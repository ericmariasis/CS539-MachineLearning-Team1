Data = imageDatastore({'/Users/seedlab/Desktop/athletic_field','/Users/seedlab/Desktop/chicken_coop/outdoor','/Users/seedlab/Desktop/basketball_court/BasketBallCourt','/Users/seedlab/Desktop/wine_cellar/bottle_storage','/Users/seedlab/Desktop/needleleaf'});
Data.Labels = zeros(size(Data.Files));
for j = 1:length(Data.Files)
    ignoreLength = length('/Users/seedlab/Desktop/');
    distinguishingCharacter = Data.Files{j}(ignoreLength+1);
    switch(distinguishingCharacter)
        case 'a'
            Data.Labels(j) = 31;
        case 'c'
            Data.Labels(j) = 42;
        case 'b'
            Data.Labels(j) = 28;
        case 'w'
            Data.Labels(j) = 25;
        case 'n'
            Data.Labels(j) = 55;
    end
   
end

[TestingData, TrainingData] = splitEachLabel(Data, 100);
[TrainingDataSample, TrainingDataTotal] = splitEachLabel(TrainingData, 100);


%net = inceptionv3
net = inceptionv3('Weights','imagenet')
%lgraph = inceptionv3('Weights','none')

analyzeNetwork(net);


net.Layers(1);
InputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);
%Freezing initial layers
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:312) = freezeWeights(layers(1:312));
lgraph = createLgraphUsingConnections(layers,connections);

[learnableLayers, classLayer] = findLayersToReplace(lgraph);
[learnableLayers, classLayer]


plot(lgraph)
title('Layer Connectivity Graph Inception V3');
FontSize = 2;

%last learnable weight layer is fully connected so we will replace this
%layer with a new fully connected layer where the number of outputs = 5
%(our number of classes)
% 

numClasses = numel(unique(TrainingDataSample.Labels));
if isa(learnableLayers,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','New Fully Connected Layer', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayers,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
else 
    error('we have no idea what is going on');
end

lgraph = replaceLayer(lgraph,learnableLayers.Name,newLearnableLayer);
plot(lgraph)
title('Layer Graph After Replacing Layers');
