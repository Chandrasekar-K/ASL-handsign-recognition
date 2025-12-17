// Webcam Access
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const captureBtn = document.getElementById("capture");
const analyzeBtn = document.getElementById("analyze");
const resultText = document.getElementById("result");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");

// Access the webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
    video.play();
  })
  .catch(err => console.error("Camera access error:", err));

// Capture Image
captureBtn.addEventListener("click", () => {
  if (video.readyState === 4) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  } else {
    console.error("Video not ready for capture.");
  }
});

// Image Upload
fileInput.addEventListener("change", event => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = e => {
      preview.src = e.target.result;
      preview.style.display = "block";
    };
    reader.readAsDataURL(file);
  }
});

// AI Integration
analyzeBtn.addEventListener("click", async () => {
  resultText.textContent = "Result: Processing...";
  
  try {
      // Get image from canvas
      const imageData = canvas.toDataURL('image/jpeg');
      
      // Send to server
      const response = await fetch('http://localhost:5000/detect', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ image: imageData })
      });
      
      if (!response.ok) throw new Error('Server error');
      
      const data = await response.json();
      
      // Display results
      if (data.result && data.result.length > 0) {
          resultText.textContent = `Result: ${data.result.join(", ")}`;
      } else {
          resultText.textContent = "No hands detected";
      }
      
      // Show annotated image
      if (data.annotated) {
          preview.src = `data:image/jpeg;base64,${data.annotated}`;
          preview.style.display = "block";
      }
      
  } catch (error) {
      console.error('Error:', error);
      resultText.textContent = "Error processing image";
  }
});
