--[[
   This file implements Variational Batch Normalization
   CAVEAT: Convolutional layers only
]]--
local VBN,parent = torch.class('nn.VariationalBatchNormalization', 'nn.Module')

function VBN:__init(nOutput)
	parent.__init(self)
	self.mean = self.mean or 0
	self.stddev = self.stddev or 0
end

function VBN:reset()
end

local function centerData(input, eps)
  -- input is a 4-tensor of data
  -- eps is a small regularizer
  eps = eps or 1e-5
  local mean = torch.mean(torch.mean(torch.mean(input, 4), 3), 1)
	--print(mean)
  local stddev = torch.repeatTensor(mean, input:size(1), 1, input:size(3), input:size(4))
  local centered = input - stddev
  stddev = torch.sqrt(torch.mean(torch.mean(torch.mean(centered, 4), 3), 1))
  local broadcastStddev = torch.repeatTensor(stddev, input:size(1), 1, input:size(3), input:size(4)) + eps
  centered:cdiv(broadcastStddev)
	return centered, mean, stddev, broadcastStddev
end

function VBN:updateOutput(input)
  -- Get central moments
  local centered, mean, stddev, broadcastStddev = centerData(input)
	self.broadcastStddev = broadcastStddev
  return centered   
end

function VBN:updateGradInput(input, gradOutput)
	local gradInputMu = 
	local gradInput = torch.div(gradOutput,broadcastStddev)
	return gradOutput
end

function VBN:accGradParameters(input, gradOutput)
  return gradOutput
end


