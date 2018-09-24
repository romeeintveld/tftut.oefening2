/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

// This is a helper class for loading and managing MNIST data specifically.
// It is a useful example of how you could create your own data manager class
// for arbitrary data though. It's worth a look :)
import {IMAGE_H, IMAGE_W, MnistData} from './data';

// This is a helper class for drawing loss graphs and MNIST images to the
// window. For the purposes of understanding the machine learning bits, you can
// largely ignore it
import * as ui from './ui';

/**
 * Creates a convolutional neural network (Convnet) for the MNIST data.
 *
 * @returns {tf.Model} An instance of tf.Model.
 */
function createConvModel() {

  //instantiate sequential model 
  const model = tf.sequential();

  //add first layer
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling'
  }));

  //add second layer
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }));

  //add remaining layers
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling'
  }));
  
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }));

  // add a flatten layer to flatten the output of the previous layer to a vector
  model.add(tf.layers.flatten());

  // add a dense layer (also known as a fully connected layer), which will perform the final classification
  model.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'VarianceScaling',
    activation: 'softmax'
  }));

  return model;
}


async function train(model) {
  ui.logStatus('Training model...');

//defining the optimizer
  const LEARNING_RATE = 0.15;
  const optimizer = tf.train.sgd(LEARNING_RATE);

  //compiling the model (+defining loss and defining the evaluation metric)
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  //how many examples the model should "see" before making a parameter update
  const BATCH_SIZE = 64;
  //how many batches to train the model for
  const TRAIN_BATCHES = 100;

  //for the test data
  const TEST_BATCH_SIZE = 1000;
  const TEST_ITERATION_FREQUENCY = 5;




  for (let i = 0; i < TRAIN_BATCHES; i++) {
    const batch = data.nextTrainBatch(BATCH_SIZE);
   
    let testBatch;
    let validationData;
    // Every few batches test the accuracy of the mode.
    if (i % TEST_ITERATION_FREQUENCY === 0) {
      testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
      validationData = [
        testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
      ];
    }
   
    // The entire dataset doesn't fit into memory so we call fit repeatedly
    // with batches.
    const history = await model.fit(
        batch.xs.reshape([BATCH_SIZE, 28, 28, 1]),
        batch.labels,
        {
          batchSize: BATCH_SIZE,
          validationData,
          epochs: 1
        });
  
    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];
  
    // ... plotting code ...
    plotAccuracy(batch, accuracy, 'train');
    plotLoss(batch, loss, 'train');
    showTestResults(batch, predictions, labels);
    draw(image, canvas);
  }

}
let data;
async function load() {
  data = new MnistData();
  await data.load();
}

// This is our main function. It loads the MNIST data, trains the model, and
// then shows what the model predicted on unseen test data.
ui.setTrainButtonCallback(async () => {
  ui.logStatus('Loading MNIST data...');
  await load();

  ui.logStatus('Creating model...');
  const model = createConvModel();//createModel();
  model.summary();

  ui.logStatus('Starting model training...');
  await train(model);

  showPredictions(model);

});
 
