require('@tensorflow/tfjs-node');

import * as tf from "@tensorflow/tfjs";


// // Define a model for linear regression.
// const model = tf.sequential();
// model.add(tf.layers.dense({units: 1, inputShape: [1]}));
//
// model.compile({loss: "meanSquaredError", optimizer: "sgd"});
//
// // Generate some synthetic data for training.
// const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
// const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
//
// // Train the model using the data.
// model.fit(xs, ys, {epochs: 10}).then(() => {
//     // Use the model to do inference on a data point the model hasn't seen before:
//     model.predict(tf.tensor2d([5], [1, 1])).print();
//     // Open the browser devtools to see the output
// });

const model = tf.sequential({
    layers: [
        tf.layers.dense({inputShape: [5], units: 1, activation: 'relu'}),
        tf.layers.dense({units: 9, activation: 'relu'}),
        tf.layers.dense({units: 7, activation: 'softmax'}),
        tf.layers.dense({units: 6, activation: 'relu'}),
        tf.layers.dense({units: 19, activation: 'relu'}),
        tf.layers.dense({units: 7, activation: 'softmax'}),
        tf.layers.dense({units: 4, activation: 'relu'}),
        tf.layers.dense({units: 1, activation: 'softmax'}),
    ]
});

model.weights.forEach(w => {
    console.log(w.name, w.shape);
});

model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError',
    metrics: ['accuracy']
});

const qqq = [
    // o     h       l       ac      v
    // [276.15, 281.09, 275.87, 277.84, 49142300], // -- sep 30
    [276.62, 277.71, 275.37, 275.95, 27050000],
    [276.58, 277.20, 271.67, 277.20, 58438300],
    [265.91, 272.40, 264.30, 271.56, 55242400],
    [261.39, 268.70, 261.22, 265.39, 70631700],
    [272.15, 272.34, 263.25, 264.16, 48837300],
    [269.99, 273.08, 266.54, 272.48, 41128900],
    [262.47, 267.65, 260.11, 267.51, 57168600],
    [271.79, 272.09, 262.63, 266.87, 86251100],
    [267.55, 274.40, 266.68, 270.32, 81570800],
    [279.88, 280.36, 274.25, 274.61, 42321000],
    [279.03, 280.45, 275.13, 279.06, 41849900],
    [274.36, 277.22, 272.96, 275.16, 38343600],
    [274.15, 275.22, 266.90, 270.45, 71750600],
    [280.94, 282.20, 270.56, 272.34, 69667500],
    [275.64, 280.05, 273.00, 277.88, 64837900],
    [272.22, 278.22, 269.66, 269.95, 99568600],
    [285.56, 288.93, 271.80, 283.58, 123852700],
    [298.10, 298.62, 284.41, 287.41, 110083300],
    [303.28, 303.50, 296.89, 302.76, 50836800],
    [297.62, 300.04, 295.79, 299.92, 36395800], // -- sep 1
    [293.07, 296.75, 292.62, 294.88, 36321000],
    [292.33, 293.18, 290.93, 292.53, 28021000],
    [292.97, 293.85, 288.71, 291.05, 49482700],
    [287.15, 292.22, 285.83, 291.96, 42216700],
    [282.83, 286.06, 282.37, 285.86, 34866600],
    [285.25, 286.00, 281.27, 283.63, 36418500],
    [280.13, 282.34, 279.46, 281.87, 34748900],
    [275.28, 280.43, 274.86, 279.93, 27723400],
    [277.93, 279.02, 275.57, 276.10, 31999400],
    [276.42, 278.46, 274.91, 277.97, 23529500],
    [273.93, 275.84, 272.20, 275.32, 24803000],
    [273.12, 273.40, 270.72, 272.16, 30799500],
    [272.66, 274.83, 271.51, 272.48, 31895700],
    [267.42, 272.84, 267.37, 271.86, 38894300],
    [269.11, 270.44, 264.63, 265.19, 43104300],
    [271.77, 271.98, 266.67, 270.31, 35088400],
    [274.08, 274.88, 269.25, 271.47, 44944900],
    [270.85, 274.98, 270.19, 274.64, 28048200],
    [270.88, 271.52, 269.96, 271.05, 21685900],
    [268.86, 270.48, 268.09, 270.38, 24911300],
    [268.00, 270.15, 267.87, 269.38, 32081600], // -- aug 3
];

//                                                                       today > tomorrow
const qqqNextDayClose = qqq.map((q, i) => i === 0 ? 1 : q[3] > qqq[i-1][3] ? 0 : 1);

console.log(qqq);
console.log(qqqNextDayClose);

function onBatchEnd(batch, logs) {
    console.log('Accuracy', logs.acc);
}

model.fit(tf.tensor2d(qqq), tf.tensor1d(qqqNextDayClose), {
    epochs: 3,
    batchSize: 3,
    callbacks: { onBatchEnd }
}).then(info => {
    console.log('Info', info);
    console.log('Final accuracy', info.history.acc);
});

setTimeout(() => {
    const prediction = model.predict(tf.tensor2d([
        [276.58, 277.20, 271.67, 277.20, 58438300], // -- sep 29
        [276.34, 280.49, 276.23, 280.16, 28184600], // -- oct 5 (buy)
        [276.02, 282.24, 273.44, 274.31, 75497400], // -- oct 2 (buy)
        [281.79, 282.88, 279.84, 282.25, 50020200], // -- oct 1 (sell)
        [276.15, 281.09, 275.87, 277.84, 49142300], // -- sep 30 (buy)
    ])) as tf.Tensor;
    prediction.print(true);


}, 2000);


tf.randomUniform([100, 10]).print();

console.log("All done")
