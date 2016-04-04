require 'nn'
require 'cunn'
require './GroupConvolutions'

-- Network-in-Network
-- achieves 92% with BN and 88% without

local model = nn.Sequential()

local function Block(...)
  local arg = {...}
  model:add(nn.GroupConvolution(...))
  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  model:add(nn.ReLU(true))
  return model
end

local function SmallBlock(...)
  local arg = {...}
  model:add(nn.SpatialConvolution(...))
  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  model:add(nn.ReLU(true))
  return model
end

--[[
model:add(nn.GroupConvolution(3,40,3,3,1,1,1,1))
model:add(nn.GroupConvolution(40,80,3,3,1,1,1,1))
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.GroupConvolution(80,80,3,3,1,1,1,1))
model:add(nn.GroupConvolution(80,40,3,3,1,1,1,1))
model:add(nn.SpatialMaxPooling(16,16,1,1):ceil())
model:add(nn.View(40))
model:add(nn.Linear(40,10))
]]--

Block(3,192,3,3,1,1,1,1)
SmallBlock(192,160,1,1,1,1,1,1)
SmallBlock(160,96,1,1,1,1,1,1)
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.Dropout())
Block(96,192,3,3,1,1,1,1)
SmallBlock(192,192,1,1,1,1,1,1)
SmallBlock(192,192,1,1,1,1,1,1)
model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout())
Block(192,192,3,3,1,1,1,1)
SmallBlock(192,192,1,1,1,1,1,1)
SmallBlock(192,12,1,1,1,1,1,1)
model:add(nn.SpatialAveragePooling(15,15,1,1):ceil())
model:add(nn.View(12))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(12,10))

return model
