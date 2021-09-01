# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:26:12 2020

@author: Argarot
"""


import pandas as pd 
from numpy import sin

from opentrons import protocol_api                                               # uncomment for experimental run

# from opentrons import simulate                                                 # uncomment for simulation
# protocol = simulate.get_protocol_api('2.7')                                    # uncomment for simulation

metadata = {
    'protocolName': 'viscous graphene dropcasting',
    'author': 'Argarot',
    'source': 'AMDM',
    'apiLevel': '2.7'
    }

def run(protocol: protocol_api.ProtocolContext):                                 # uncomment for experimental run
  protocol.home()    
    
  # =============================================================================
  #   define Labware, pipettes and tipracks
  # =============================================================================
  tiprack300_1 = protocol.load_labware('opentrons_96_tiprack_300ul', 8)
  tiprack300_2 = protocol.load_labware('opentrons_96_tiprack_300ul', 11)
  p300 = protocol.load_instrument('p300_single', mount='right', tip_racks=[tiprack300_1, tiprack300_2])  

  stock_solutions  =  protocol.load_labware('amdm_12_40ml_vial_adapter', 4)
  mixing_plate = protocol.load_labware('corning_96_wellplate_360ul_flat', 1)
  dropcast_plate_1 = protocol.load_labware('amdm_49_wellplate_to_wafer_20ul', 2)
  dropcast_plate_2 = protocol.load_labware('amdm_49_wellplate_to_wafer_20ul', 9)
  # 49 (7x7) grid was chosen because 64 (8x8) is too dense and droplets keep merging.
  # but 49 is exactly sufficient to fit full 96-wellplate worth of samples on 2 wafers
  
  
  # =============================================================================
  #   read the instructions from the .CSV file
  # =============================================================================   
  df = pd.read_csv(r'/data/user_storage/DoE_v3_PVP.csv', index_col=0)    # uncomment for experimental run
  # df = pd.read_csv('DoE_v2.csv', index_col=0)                              # uncomment for simulation
  mix_positions = df.values[:,7]
  dropcast_positions = df.values[:,8]
  wafers = df.values[:,9]
  
  solutions = {'HPMC':     {'stock_position': stock_solutions['C3'], # user-defined position of HPMC stock solution
                            'amount':         df.values[:,6]                              
                            },
               'surf':     {'stock_position': stock_solutions['B2'], # user-defined position of surfactant stock solution
                            'amount':         df.values[:,5]                              
                            },
               'graphene': {'stock_position': stock_solutions['A1'], # user-defined position of graphene stock solution
                            'amount':         df.values[:,4]                               
                            },              
               }  
    
  p300.default_speed = 250 # slow down the robot to reduce chance of losing droplets while moving
  
  
  # =============================================================================
  #   distribute viscous HPMC
  # =============================================================================
  
  gap = 51                 # distance from the top of the well, in mm, at which to aspirate HPMC        <<< ADJUST
  p300.pick_up_tip()
  
  for well, vol in zip(mixing_plate.wells(), list(solutions['HPMC']['amount'])):
    p300.aspirate(vol, solutions['HPMC']['stock_position'].top(-gap))
    protocol.delay(5)      # wait for the HPMC to flow in the pipette
    p300.touch_tip()       # get rid of the drop hanging on the outside
    p300.dispense(vol, well, 50)
    protocol.delay(3)      # wait for HPMC to flow out of the tip
    gap += vol*0.0018      # increase the gap by the amount aspirated. keeps depth of aspiration constant
  
  p300.drop_tip()
  
  
# =============================================================================
#   distribute rest of solutions
# =============================================================================
  
  for solution in ['surf','graphene']:
    
    p300.distribute(list(solutions[solution]['amount']), 
                    solutions[solution]['stock_position'], 
                    [well.top(-1) for well in mixing_plate.wells()[0:len(solutions[solution]['amount'])]], 
                    # dispense 1mm below the top of the well to avoid cross-contamination
                    new_tip = 'once', # use one pipette per stock solution
                    trash = True,     # trash tip when done
                    touch_tip = True) # get rid of the drop hanging on the outside


# =============================================================================
#   dropcasting
# =============================================================================
  
  dcvol = 11  # dropcasting volume, ul. More - droplets merge, less - hard to characterize
  
  p300.pick_up_tip()
  
  for mix_pos, wafer, dropcast_position, j in zip(list(mix_positions), wafers, list(dropcast_positions), range(len(mix_positions))):
    
    if j < 0: # for calibration. number of wells to skip
      continue
    else:

      if wafer == 1:
        destination = dropcast_plate_1
      else:
        destination = dropcast_plate_2
      
      # mixing:================================================================
      for i in range(1,13):    # repeat 12 times
        p300.aspirate(50, mixing_plate[mix_pos].bottom(0.5+sin(i*15)/10))
        # aspirate 50ul at 0.5+/-0.1 mm distance from the bottom (varying height with sin() function for better mixing)
        protocol.delay(3)      # wait for the HPMC to flow in the pipette
        p300.dispense(250, mixing_plate[mix_pos].bottom(2+i/4))
        # dipense (more than aspirate() allows smaller delay) at varying height for better mixing
        protocol.delay(2)      # wait for HPMC to flow out of the tip
      
      # dropcasting:===========================================================
      p300.aspirate(dcvol, mixing_plate[mix_pos].bottom(0.4))              
      protocol.delay(4)      # wait for the HPMC to flow in the pipette
      p300.dispense(dcvol+dcvol*0.1, destination[dropcast_position].bottom(-0.1)) 
      # dispense as close to the wafer as possible. most of calibration went here
      protocol.delay(4)      # wait for HPMC to flow out of the tip
      p300.air_gap(50, 5)    # instert air gap for safer transport
      #p300.return_tip()
      
      # washing the tip:=======================================================
      for well in ['A2','A3','A4','B3','B4']:
        p300.mix(2, 250, stock_solutions[well].top(-50), 120)
        p300.blow_out()
      p300.touch_tip()  

