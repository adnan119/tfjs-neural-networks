const model = tf.sequential();

model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.add(tf.layers.dense({units: 18, activation: 'sigmoid'}));
model.add(tf.layers.dense({units: 1}));
model.compile({loss:'meanSquaredError',

optimizer:'adam'});

model.summary();

const xs = tf.tensor2d([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10,1]);
const ys = tf.tensor2d([ 5.0, 2.0, 1.0, 2.0, 5.0, 10.0, 17.0, 26.0, 37.0, 50.0], [10,1]);

async function doTraining(model){
    const history = 
    await model.fit(xs ,ys,
        { epochs:2500,
        callbacks:{
            onEpochEnd: async(epoch, logs) =>{
                console.log("Epochs: " + epoch + " Loss:" + logs.loss);
            }
        }
    });
}

doTraining(model).then(() => {
    alert(model.predict(tf.tensor2d([10], [1,1])));
});