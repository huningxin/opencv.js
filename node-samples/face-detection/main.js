let v4l2camera = require( "v4l2camera" );
var jpeg = require('jpeg-js');
var fs = require('fs');
let cv = require('opencv.js');

cv.FS_createLazyFile('/', 'haarcascade_frontalface_default.xml',
                         'haarcascade_frontalface_default.xml', true, false);
let faceClassifier = new cv.CascadeClassifier();
faceClassifier.load('haarcascade_frontalface_default.xml');

console.log('OpenCV cascade classifier created')
 
let cam = new v4l2camera.Camera("/dev/video0");
if (cam.configGet().formatName !== "YUYV") {
    console.log("YUYV camera required");
    process.exit(1);
}
cam.configSet({width: 320, height: 240});
let format = cam.configGet();
console.log("Camera config [ " + format.formatName + " " + format.width + "x" + 
    format.height + " " + format.interval.numerator + "/" + 
    format.interval.denominator + "]");

cam.start();
console.log('Start camera...');

let yuvMat = null;
let rgbMat = null;
let grayMat = null;

let stopped = false;
let frameIndex = 0;

cam.capture(function detectFace(success) {
  let frame = cam.frameRaw();
  let videoHeight = cam.height;
  let videoWidth = cam.width;
  if (!yuvMat)
    yuvMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC2);
  yuvMat.data.set(frame);
  if (!rgbMat)
    rgbMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC4);
  cv.cvtColor(yuvMat, rgbMat, cv.COLOR_YUV2RGBA_YUYV);
  if (!grayMat)
    grayMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC1);
  cv.cvtColor(rgbMat, grayMat, cv.COLOR_RGBA2GRAY);

  let faces = [];
  let eyes = [];
  let size;
  let faceVect = new cv.RectVector();
  let faceMat = new cv.Mat();

  cv.pyrDown(grayMat, faceMat);
  if (videoWidth > 320)
    cv.pyrDown(faceMat, faceMat);
  size = faceMat.size();

  console.log('Process frame ' + frameIndex++);
  faceClassifier.detectMultiScale(faceMat, faceVect);
  for (let i = 0; i < faceVect.size(); i++) {
    let xRatio = videoWidth/size.width;
    let yRatio = videoHeight/size.height;
    let face = faceVect.get(i);
    let x = face.x*xRatio;
    let y = face.y*yRatio;
    let w = face.width*xRatio;
    let h = face.height*yRatio;
    let point1 = new cv.Point(x, y);
    let point2 = new cv.Point(x + w, y + h);
    cv.rectangle(rgbMat, point1, point2, [255, 0, 0, 255]);
    console.log('\tFace detected : ' + '[' + i + ']' +
        ' (' + x + ', ' + y + ', ' + w + ', ' + h + ')');
  }
  faceMat.delete();
  faceVect.delete();
  if (stopped) {
    cam.stop();
    console.log('Stopped');
    rawData = {
      data: rgbMat.data,
      width: rgbMat.size().width,
      height: rgbMat.size().height
    };
    var jpegData = jpeg.encode(rawData, 50);
    const filename = 'result.jpg';
    fs.writeFileSync(filename, jpegData.data);
    console.log('Written into ' + filename);
    yuvMat.delete();
    rgbMat.delete();
    grayMat.delete();
    process.exit();
  }
  cam.capture(detectFace);
});

const ESC_KEY = '\u001b';
const CTRL_C = '\u0003';
let stdin = process.stdin;
stdin.setRawMode(true);
stdin.resume();
stdin.setEncoding('utf8');
stdin.on('data', function(key) {
  if (key === ESC_KEY || key === CTRL_C) {
    stopped = true;
  }
});
