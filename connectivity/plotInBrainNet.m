function [] = plotInBrainNet(edgepath, nodepath, varargin)

iP = inputParser;
addParameter(iP, 'options', []);
addParameter(iP, 'surface', '/path/to/brainnet/BrainMesh.nv');
parse(iP, varargin{:});

% setup the parameters that will go into BrainNetViewer
brainnet_params = {iP.Results.surface, edgepath, nodepath};
if ~isempty(iP.Results.options)
  brainnet_params{end+1} = iP.Results.options;
end

% call BrainNetViewer
BrainNet_MapCfg(brainnet_params{:});
end