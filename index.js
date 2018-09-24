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
import {IMAGE_H, IMAGE_W, MnistData} from './data';
import * as ui from './ui';


function createConvModel() {

  // Instantiate sequential model  
  // tf.sequential provides an API for creating "stacked" models where the output from 
  // one layer is used as the input to the next layer
   const model = tf.sequential();

  // Add the first layer
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_H, IMAGE_W, 1],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));

  // Add the second layer
  model.add(tf.layers.maxPooling2d({
    poolSize: 2, 
    strides: 2
  }));

  // Add the third layer
  // This is another convolution, this time with 32 filters instead of 16
  model.add(tf.layers.conv2d({
    kernelSize: 3, 
    filters: 32, 
    activation: 'relu'
  }));

  // The fourth layer
  // Max pooling again.
  model.add(tf.layers.maxPooling2d({
    poolSize: 2, 
    strides: 2
  }));

  // Add another conv2d layer
  model.add(tf.layers.conv2d({
    kernelSize: 3, 
    filters: 32, 
    activation: 'relu'
  }));

  // Add a flatten layer to flatten the output of the previous layer to a vector
  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({
    units: 64, 
    activation: 'relu'
  }));

   // add a dense layer (also known as a fully connected layer), 
   //which will perform the final classification
  model.add(tf.layers.dense({
    units: 10, 
    activation: 'softmax'
  }));

  return model;
}




/**
 * Creates a model consisting of only flatten, dense and dropout layers.
 *
 * The model create here has approximately the same number of parameters
 * (~31k) as the convnet created by `createConvModel()`, but is
 * expected to show a significantly worse accuracy after training, due to the
 * fact that it doesn't utilize the spatial information as the convnet does.
 *
 * This is for comparison with the convolutional network above.
 *
 * @returns {tf.Model} An instance of tf.Model.
 */
function createDenseModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape: [IMAGE_H, IMAGE_W, 1]}));
  model.add(tf.layers.dense({units: 42, activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  return model;
}

/**
 * Compile and train the given model.
 *
 * @param {*} model The model to
 */
async function train(model) {
  ui.logStatus('Training model...');

  // Defining the optimizer
  const LEARNING_RATE = 0.01;
  const optimizer = 'rmsprop';

  // Compiling the model (+defining loss and defining the evaluation metric)
  // We compile our model by specifying an optimizer, a loss function, and a
  // list of metrics that we will use for model evaluation
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });


  // How many examples the model should "see" before making a parameter update 
  const batchSize = 64;

  // Leave out the last 15% of the training data for validation, to monitor
  // overfitting during training
  const validationSplit = 0.15;

  // Get number of training epochs from the UI
  const trainEpochs = ui.getTrainEpochs();

  // Keep a buffer of loss and accuracy values over time
  let trainBatchCount = 0;

  const trainData = data.getTrainData();
  const testData = data.getTestData();

  const totalNumBatches =
      Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) *
      trainEpochs;

  // During the long-running fit() call for model training, we include
  // callbacks, so that we can plot the loss and accuracy values in the page
  // as the training progresses.
  let valAcc;
  await model.fit(trainData.xs, trainData.labels, {
    batchSize,
    validationSplit,
    epochs: trainEpochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainBatchCount++;
        ui.logStatus(
            `Training... (` +
            `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
            ` complete). To stop training, refresh or close page.`);
        ui.plotLoss(trainBatchCount, logs.loss, 'train');
        ui.plotAccuracy(trainBatchCount, logs.acc, 'train');
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        valAcc = logs.val_acc;
        ui.plotLoss(trainBatchCount, logs.val_loss, 'validation');
        ui.plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
        await tf.nextFrame();
      }
    }
  });

  const testResult = model.evaluate(testData.xs, testData.labels);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  const finalValAccPercent = valAcc * 100;
  ui.logStatus(
      `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
      `Final test accuracy: ${testAccPercent.toFixed(1)}%`);
}

/**
 * Show predictions on a number of test examples.
 *
 * @param {tf.Model} model The model to be used for making the predictions.
 */
async function showPredictions(model) {
  const testExamples = 100;
  const examples = data.getTestData(testExamples);

  // Code wrapped in a tf.tidy() function callback will have their tensors freed
  // from GPU memory after execution without having to call dispose().
  // The tf.tidy callback runs synchronously.
  tf.tidy(() => {
    const output = model.predict(examples.xs);

    // tf.argMax() returns the indices of the maximum values in the tensor along
    // a specific axis. Categorical classification tasks like this one often
    // represent classes as one-hot vectors. One-hot vectors are 1D vectors with
    // one element for each output class. All values in the vector are 0
    // except for one, which has a value of 1 (e.g. [0, 0, 0, 1, 0]). The
    // output from model.predict() will be a probability distribution, so we use
    // argMax to get the index of the vector element that has the highest
    // probability. This is our prediction.
    // (e.g. argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)
    // dataSync() synchronously downloads the tf.tensor values from the GPU so
    // that we can use them in our normal CPU JavaScript code
    // (for a non-blocking version of this function, use data()).
    const axis = 1;
    const labels = Array.from(examples.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());

    ui.showTestResults(examples, predictions, labels);
  });
}

function createModel() {
  let model;
  const modelType = ui.getModelTypeId().split(" ")[0];
  if (modelType === 'ConvNet') {
    model = createConvModel();
  } else if (modyarelType === 'DenseNet') {
    model = createDenseModel();
  } else {
    throw new Error(`Invalid model type: ${modelType}`);
  }
  return model;
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
  const model = createModel();
  model.summary();

  ui.logStatus('Starting model training...');
  await train(model);

  showPredictions(model);
});