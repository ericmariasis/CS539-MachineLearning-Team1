Data = imageDatastore({'/Users/seedlab/Desktop/athletic_field','/Users/seedlab/Desktop/chicken_coop/outdoor','/Users/seedlab/Desktop/basketball_court/BasketBallCourt','/Users/seedlab/Desktop/wine_cellar/bottle_storage','/Users/seedlab/Desktop/needleleaf'});
s = zeros(size(Data.Files));
for j = 1:length(Data.Files)
    ignoreLength = length('/Users/seedlab/Desktop/');
    distinguishingCharacter = Data.Files{j}(ignoreLength+1);
    switch(distinguishingCharacter)
        case 'a'
            s(j) = 31;
        case 'c'
            s(j) = 42;
        case 'b'
            s(j) = 28;
        case 'w'
            s(j) = 25;
        case 'n'
            s(j) = 55;
        otherwise
            error(['Probelm ', num2str(j)]);
    end
   
end
if any(s==0)
    error('We didn''t assign nuthin');
end
Data.Labels = categorical(s);
[TestingData, TrainingData] = splitEachLabel(Data, 100);
[TrainingDataSample, TrainingDataTotal] = splitEachLabel(TrainingData, 100);


%net = inceptionv3
net = inceptionv3('Weights','imagenet')
%lgraph = inceptionv3('Weights','none')

analyzeNetwork(net);


net.Layers(1);
InputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);

[learnableLayers, classLayer] = findLayersToReplace(lgraph);
[learnableLayers, classLayer]


plot(lgraph)
title('Layer Connectivity Graph Inception V3');
FontSize = 2;

%last learnable weight layer is fully connected so we will replace this
%layer with a new fully connected layer where the number of outputs = 5
%(our number of classes)

numClasses = 5;
if isa(learnableLayers,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','New Fully Connected Layer', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayers,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dlayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
else 
    error('we have no idea what is going on');
end

lgraph = replaceLayer(lgraph,learnableLayers.Name,newLearnableLayer);
plot(lgraph)
title('Layer Graph After Replacing Layers');

newClassificationLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph, classLayer.Name, newClassificationLayer);

figure('Units','Normalized','Position', [0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%Freezing initial layers
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:312) = freezeWeights(layers(1:312));
lgraph = createLgraphUsingConnections(layers,connections);
figure;
plot(lgraph)
title('Layers after freezing');
%Training the 
imageSize = [299 299 3];

trainingauimds = augmentedImageDatastore(imageSize, TrainingDataSample);
testingauimds = augmentedImageDatastore(imageSize, TestingData);
miniBatchSize = 10;


options = trainingOptions('sgdm', 'MiniBatchSize', miniBatchSize, 'MaxEpochs', 6, ...
    'InitialLearnRate', 3e-4, 'Shuffle', 'every-epoch','Verbose', false ,'Plots', 'training-progress');
net = trainNetwork(trainingauimds,lgraph,options);

[ClassPredict,ActualClass] = classify(net,testingauimds);
accuracy = mean(ClassPred == ActualClass.Labels);
