let testInfo = document.getElementById("testInfo");
let iterationProgress = document.getElementById("iterationProgress");
let showIterations = document.getElementById("showIterations");
let modelLoad = document.getElementById("modelLoad");
let modelProgress = document.getElementById("modelProgress");
let showModel = document.getElementById("showModel");

let runButton = document.getElementById("runButton");
runButton.addEventListener("click", run);
// Run test only after opencv.js loaded
cv.onRuntimeInitialized = () => {
    runButton.disabled = false;
}

let imgelement = document.getElementById("imgsrc");
let inputelement = document.getElementById("fileinput");
inputelement.addEventListener("change", (e) =>{
    imgelement.src = URL.createObjectURL(e.target.files[0]);
}, false);

const modelZoo = [
    {name: 'squeezenet', url: 'https://webnnmodel.s3-us-west-2.amazonaws.com/image_classification/model/squeezenet1.1.onnx'},
    {name: 'mobilenetv2', url: 'https://webnnmodel.s3-us-west-2.amazonaws.com/image_classification/model/mobilenetv2-1.0.onnx'},
    {name: 'resnet50v1', url: 'https://webnnmodel.s3-us-west-2.amazonaws.com/image_classification/model/resnet50v1.onnx'},
    {name: 'resnet50v2', url: 'https://webnnmodel.s3-us-west-2.amazonaws.com/image_classification/model/resnet50v2.onnx'}
];

/*************************   CONTROL PARAMETERS   **************************/

// Record if the model have been loaded
let modelStatus = ['squeezenet', 'mobilenetv2', 'resnet50v1', 'resnet50v2'];
// The forward iterations set by user
let iterations = Number(document.querySelector('#iterations').value);
// Flag for first run
let initFlag = true;
// Count for iterations finished
let calIteration = 0;
// Number of top result to show
let topNum = 5;
// Save each forward time 
let timeSum = [];
// Lables for image classification result
let labels;
// Top result to show
let classes;

//Detect the click, init the UI and control parameters,
//check if the model have been loaded, then run the test.
function run(){
    initPara();
    let onnxmodel = document.getElementById("modelName").value;
    let index = modelStatus.indexOf(onnxmodel);
    if ( index != -1){
        modelInfo = getModelById(onnxmodel);
        modelUrl = modelInfo.url;
        onnxmodel += '.onnx';
        showModel.innerHTML = 'Model loading...';
        createFileFromUrl(onnxmodel, modelUrl, compute, modelState);
        modelStatus.splice(index, 1);
    } else{
        modelLoad.innerHTML = `${onnxmodel} has been loaded before.`;
        compute();
    };
}

function getModelById(id){
    for(const modelInfo of modelZoo){
        if (id === modelInfo.name) {
            return modelInfo;
        };
    };
    return {};
}

//Init the UI and the control parameters.
function initPara(){
    iterations = Number(document.querySelector('#iterations').value);
    calIteration = 0;
    timeSum = [];

    testInfo.innerHTML = '';
    modelLoad.innerHTML = '';
    showModel.innerHTML = '';
    modelProgress.value = 0;
    modelProgress.style.visibility = 'hidden';
    iterationProgress.style.visibility = 'hidden';
    showIterations.innerHTML = '';

    if(initFlag){
        loadLables();
        initFlag = false;
    };
}

//Load labels from the txt for the first run.
function loadLables(){
    let url = 'labels1000.txt';
    let request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.onload = function(ev) {
        if (request.readyState ===4 ) {
            if(request.status === 200) {
                labels = request.response;
                labels = labels.split('\n')
            };
        };
    };
    request.send();
}

//The whole compute pipeline.
function compute (){
    let inputMat = imageToMat();
    let onnxmodel = document.getElementById("modelName").value + '.onnx';
    let net = cv.readNetFromONNX(onnxmodel);
    console.log('Start inference...')
    let input = cv.blobFromImage(inputMat, 1, new cv.Size(224, 224), new cv.Scalar(0,0,0));
    net.setInput(input);

    eachForward(net);
}

//Excute forward function one by one, update the UI during each forward.
async function eachForward(net){
    let start = performance.now();
    let result =await excute(net);
    let end = performance.now();

    classes = getTopClasses(result);
    let delta = end - start;
    console.log(`Iterations: ${calIteration+1} / ${iterations+1}, inference time: ${delta}ms`);
    printResult(classes);
    timeSum.push(delta);   
    iterationState();
    ++calIteration;

    if(calIteration<iterations+1){
        setTimeout(function(){
            eachForward(net);
        }, 0);
    } else{
        console.log('Test finished!');
        updateResult(classes);
        net.delete();
        result.delete();
    };
}

//Excute the net
async function excute(net){
    let result = net.forward();
    return result;
}

//Read the image from webpage, do the image processing.
function imageToMat(){
    let mat = cv.imread("imgsrc");
    let matC3 = new cv.Mat(mat.matSize[0],mat.matSize[1],cv.CV_8UC3);
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2RGB);
    let matdata = matC3.data;
    let stddata = [];
    for(var i=0; i<mat.matSize[0]*mat.matSize[1]; ++i){
        stddata.push( (matdata[3*i]/255-0.485)/0.229 ); 
        stddata.push( (matdata[3*i+1]/255-0.456)/0.224 ); 
        stddata.push( (matdata[3*i+2]/255-0.406)/0.225 );
    };
    let inputMat = cv.matFromArray(mat.matSize[0],mat.matSize[1],cv.CV_32FC3,stddata);

    return inputMat;
}

//Update the iteration UI during the compute process.
function iterationState() {
    iterationProgress.style.visibility = 'visible';
    iterationProgress.value = (calIteration)*100/(iterations);
    showIterations.innerHTML = `Iterations: ${calIteration} / ${iterations}`;
}

//Show the final result in the webpage.
function updateResult(classes){
    let finalResult = summarize(timeSum);
    testInfo.style.visibility="visible";
    testInfo.innerHTML = `<b>Build type</b>: ${document.getElementById("title").innerHTML.split(/[()]/)[1]} <br>
                                                    <b>Model</b>: ${document.getElementById("modelName").value} <br>
                                                    <b>Inference Time</b>: ${finalResult.mean.toFixed(2)}`;
    if(iterations != 1){
        testInfo.innerHTML += ` ± ${finalResult.std.toFixed(2)} [ms] <br> <br>`;
    } else{
        testInfo.innerHTML += ` [ms] <br> <br>`;
    };
    testInfo.innerHTML += `<b>label1</b>: ${classes[0].label}, probability: ${classes[0].prob}% <br>
                           <b>label2</b>: ${classes[1].label}, probability: ${classes[1].prob}% <br>
                           <b>label3</b>: ${classes[2].label}, probability: ${classes[2].prob}% <br>
                           <b>label4</b>: ${classes[3].label}, probability: ${classes[3].prob}% <br>
                           <b>label5</b>: ${classes[4].label}, probability: ${classes[4].prob}%` ;
}

//Load the file from the filesystem.
function createFileFromUrl(path, url, callback, onprogress){
    let request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.responseType = 'arraybuffer';
    request.onload = function(ev) {
        if (request.readyState === 4) {
            if (request.status === 200) {
                let data = new Uint8Array(request.response);
                cv.FS_createDataFile('/', path, data, true, false, false);
                showModel.innerHTML = 'Model loaded.';
                callback();
            } else {
                console.log('Failed to load ' + url + ' status: ' + request.status);
            }
        }
    };
    request.send();
    request.onprogress = onprogress;
};

//Show the model status when load the model.
function modelState(ev){
    modelProgress.style.visibility = 'visible';
    let totalSize = ev.total / (1000 * 1000);
    let loadedSize = ev.loaded / (1000 * 1000);
    let percentComplete = ev.loaded / ev.total * 100;
    modelLoad.innerHTML = `${loadedSize.toFixed(2)}/${totalSize.toFixed(2)}MB ${percentComplete.toFixed(2)}%`;
    modelProgress.value = percentComplete;
}

//Add softmax layer to gengerate the probility.
function softmax(arr) {
    const C = Math.max(...arr);
    const d = arr.map((y) => Math.exp(y - C)).reduce((a, b) => a + b);
    return arr.map((value, index) => { 
        return Math.exp(value - C) / d;
    })
}

//Find the top num labels from the forward result.
function getTopClasses(mat) {
    let initdata = mat.data32F;
    initdata = softmax(initdata);
    let probs = Array.from(initdata);
    let indexes = probs.map((prob, index) => [prob, index]);
    let sorted = indexes.sort((a, b) => {
    if (a[0] === b[0]) {return 0;}
    return a[0] < b[0] ? -1 : 1;
    });
    sorted.reverse();
    let classes = [];
    for (let i = 0; i < topNum; ++i) {
    let prob = sorted[i][0];
    let index = sorted[i][1];
    let c = {
        label: labels[index],
        prob: (prob * 100).toFixed(2)
    }
    classes.push(c);
    }
    return classes;
}

//Print each inference result in the console.
function printResult(classes){
    for (let i = 0; i < topNum; ++i){
        console.log(`label: ${classes[i].label}, probability: ${classes[i].prob}%`)
    }
}

//Generate the summarize result from the collect time.
function summarize(results) {
    if (results.length !== 0) {
        // remove first run, which is regarded as "warming up" execution
        results.shift();
        let d = results.reduce((d, v) => {
            d.sum += v;
            d.sum2 += v * v;
            return d;
        }, {
            sum: 0,
            sum2: 0
        });
        let mean = d.sum / results.length;
        let std = Math.sqrt((d.sum2 - results.length * mean * mean) / (results.length - 1));
        return {
            mean: mean,
            std: std
        };
    } else {
        return null;
    };
}
