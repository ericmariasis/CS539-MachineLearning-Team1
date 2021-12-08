
TrainingData = readtable('C:\Users\Katy\Desktop\train_places205.csv');
TestingData = readtable('C:\Users\Katy\Desktop\val_places205.csv');


%net = inceptionv3
net = inceptionv3('Weights','imagenet')
%lgraph = inceptionv3('Weights','none')

%analyzeNetwork(net);

net.Layers(1);
InputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);
[learnableLayers, classLayer] = findLayersToReplace(lgraph);
[learnableLayers, classLayer]
lgraph = removeLayers(lgraph, {'predictions','predictions_softmax','ClassificationLayer_predictions'});


plot(lgraph)
title('Layer Connectivity Graph Inception V3');
FontSize = 2;

%last learnable weight layer is fully connected so we will replace this
%layer with a new fully connected layer where the number of outputs = 5
%(our number of classes)

%numClasses = numel(categories(TrainingData.Labels));
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

%lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

plot(lgraph)
title('Layer Graph After Replacing Layers');

%Freezing initial layers
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:312) = freezeWeights(layers(1:312));
lgraph = createLgraphUsingConnections(layers,connections);

