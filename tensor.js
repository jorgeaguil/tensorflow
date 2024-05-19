async function loadModelAndPredict() {
  // Load MobileNet model
  const model = await mobilenet.load()

  // Get the uploaded image element
  const imgElement = document.getElementById('image-preview')

  // Pre-process the image
  const tfImg = tf.browser
    .fromPixels(imgElement)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .expandDims()

  // Perform inference
  const predictions = await model.classify(tfImg)

  // Display predictions
  const predictionDiv = document.getElementById('prediction')
  predictionDiv.innerHTML = ''
  predictions.forEach((prediction) => {
    predictionDiv.innerHTML += `${prediction.className}: ${Math.round(
      prediction.probability * 100
    )}%<br>`
  })
}

function previewImage(event) {
  const input = event.target
  const reader = new FileReader()
  reader.onload = function () {
    const imgElement = document.getElementById('image-preview')
    imgElement.src = reader.result
    loadModelAndPredict()
  }
  reader.readAsDataURL(input.files[0])
}
