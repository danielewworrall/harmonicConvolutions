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

Block(3,192,3,3,1,1,1,1)
Block(192,160,3,3,1,1,1,1)
Block(160,96,3,3,1,1,1,1)
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
Block(96,192,3,3,1,1,1,1)
Block(192,192,3,3,1,1,1,1)
Block(192,192,3,3,1,1,1,1)
model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
Block(192,192,3,3,1,1,1,1)
Block(192,192,3,3,1,1,1,1)
model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())
model:add(nn.Dropout(0.5))
model:add(nn.View(192))
model:add(nn.Linear(192,10))



return model
