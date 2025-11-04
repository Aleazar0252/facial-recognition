let stream = null;
let continuousInterval = null;
let isContinuousActive = false;

document.addEventListener('DOMContentLoaded', function() {
    const startCameraBtn = document.getElementById('startCamera');
    const captureBtn = document.getElementById('capture');
    const retakeBtn = document.getElementById('retake');
    const recognizeBtn = document.getElementById('recognize');
    const startContinuousBtn = document.getElementById('startContinuous');
    const stopContinuousBtn = document.getElementById('stopContinuous');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const capturedImage = document.getElementById('capturedImage');
    const imageUpload = document.getElementById('imageUpload');
    const noCamera = document.getElementById('noCamera');

    startCameraBtn.addEventListener('click', startCamera);
    captureBtn.addEventListener('click', capturePhoto);
    retakeBtn.addEventListener('click', retakePhoto);
    recognizeBtn.addEventListener('click', recognizeFace);
    startContinuousBtn.addEventListener('click', startContinuousRecognition);
    stopContinuousBtn.addEventListener('click', stopContinuousRecognition);
    imageUpload.addEventListener('change', handleImageUpload);

    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user' }
            });
            video.srcObject = stream;
            video.classList.remove('d-none');
            startCameraBtn.classList.add('d-none');
            captureBtn.classList.remove('d-none');
            startContinuousBtn.classList.remove('d-none');
            noCamera.classList.add('d-none');
        } catch (err) {
            console.error('Error accessing camera:', err);
            noCamera.classList.remove('d-none');
            startCameraBtn.classList.add('d-none');
        }
    }

    function capturePhoto() {
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }

        capturedImage.src = canvas.toDataURL('image/png');
        capturedImage.classList.remove('d-none');
        video.classList.add('d-none');
        captureBtn.classList.add('d-none');
        startContinuousBtn.classList.add('d-none');
        retakeBtn.classList.remove('d-none');
        recognizeBtn.classList.remove('d-none');
    }

    function retakePhoto() {
        capturedImage.classList.add('d-none');
        retakeBtn.classList.add('d-none');
        recognizeBtn.classList.add('d-none');
        startContinuousBtn.classList.remove('d-none');
        startCamera();
    }

    async function recognizeFace() {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '<div class="alert alert-info">Recognizing faces...</div>';

        try {
            canvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append('image', blob, 'capture.jpg');

                const response = await fetch('/api/recognize', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                displayResults(data);
            }, 'image/jpeg');
        } catch (error) {
            resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        }
    }

    async function startContinuousRecognition() {
        if (isContinuousActive) return;

        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const resultsDiv = document.getElementById('results');

        if (!stream) {
            alert('Please start camera first');
            return;
        }

        isContinuousActive = true;
        resultsDiv.innerHTML = '<div class="alert alert-info recognition-active">Continuous recognition active...</div>';
        document.getElementById('startContinuous').classList.add('d-none');
        document.getElementById('stopContinuous').classList.remove('d-none');
        document.getElementById('capture').classList.add('d-none');

        continuousInterval = setInterval(async () => {
            try {
                const context = canvas.getContext('2d');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('image', blob, 'frame.jpg');

                    const response = await fetch('/api/continuous_recognition/frame', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    if (data.success) {
                        displayContinuousResults(data.results);
                    }
                }, 'image/jpeg');

            } catch (error) {
                console.error('Continuous recognition error:', error);
            }
        }, 500);
    }

    function stopContinuousRecognition() {
        if (!isContinuousActive) return;

        isContinuousActive = false;
        if (continuousInterval) {
            clearInterval(continuousInterval);
            continuousInterval = null;
        }

        document.getElementById('startContinuous').classList.remove('d-none');
        document.getElementById('stopContinuous').classList.add('d-none');
        document.getElementById('capture').classList.remove('d-none');

        document.getElementById('results').innerHTML = '<div class="alert alert-warning">Continuous recognition stopped</div>';
    }

    function displayContinuousResults(results) {
        const resultsDiv = document.getElementById('results');

        if (results.length === 0) {
            resultsDiv.innerHTML = '<div class="alert alert-warning">No faces detected</div>';
            return;
        }

        let html = '<h4>Live Recognition:</h4>';
        results.forEach((result, index) => {
            const confidenceClass = result.confidence > 70 ? 'face-known' : 'face-unknown';
            const livenessClass = result.is_live ? 'text-live' : 'text-suspicious';
            const livenessText = result.is_live ? 'Live' : 'Suspicious';
            const livenessBorder = result.is_live ? 'face-live' : 'face-suspicious';

            html += `
                <div class="face-result ${confidenceClass} ${livenessBorder} p-3 mb-2">
                    <h5>Face ${index + 1} <small class="${livenessClass}">(${livenessText})</small></h5>
                    <p><strong>Name:</strong> ${result.name}</p>
                    <p><strong>Confidence:</strong> ${result.confidence}%</p>
                    <p><strong>Liveness Score:</strong> ${result.liveness_score}</p>
                    <p><strong>Location:</strong> Top: ${result.location.top}, Left: ${result.location.left}</p>
                    <p><small class="text-muted">ID: ${result.face_id}</small></p>
                </div>
            `;
        });

        resultsDiv.innerHTML = html;
    }

    async function handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '<div class="alert alert-info">Recognizing faces...</div>';

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/api/recognize', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        }
    }

    function displayResults(data) {
        const resultsDiv = document.getElementById('results');
        
        if (!data.success) {
            resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
            return;
        }

        if (data.results.length === 0) {
            resultsDiv.innerHTML = '<div class="alert alert-warning">No faces detected in the image.</div>';
            return;
        }

        let html = '<h4>Recognition Results:</h4>';
        data.results.forEach((result, index) => {
            const confidenceClass = result.confidence > 70 ? 'face-known' : 'face-unknown';
            html += `
                <div class="face-result ${confidenceClass} p-3 mb-2">
                    <h5>Face ${index + 1}</h5>
                    <p><strong>Name:</strong> ${result.name}</p>
                    <p><strong>Confidence:</strong> ${result.confidence}%</p>
                    <p><strong>Location:</strong> Top: ${result.location.top}, Left: ${result.location.left}</p>
                </div>
            `;
        });

        resultsDiv.innerHTML = html;
    }
});