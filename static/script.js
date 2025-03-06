function openModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.style.display = 'flex';

    if (modalId === 'registerModal' || modalId === 'recognizeModal') {
        const videoId = modalId === 'registerModal' ? 'registerVideo' : 'recognizeVideo';
        initCamera(document.getElementById(videoId));
    }

    if (modalId === 'deleteModal') {
        loadPersonList();
    }

    if (modalId === 'listModal') {
        openListModal();
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.style.display = 'none';

    if (modalId === 'registerModal' || modalId === 'recognizeModal') {
        const videoId = modalId === 'registerModal' ? 'registerVideo' : 'recognizeVideo';
        const videoElement = document.getElementById(videoId);
        const stream = videoElement.srcObject;
        
        if (stream) {
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            videoElement.srcObject = null;
        }
    }
}


async function initCamera(videoElement) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
    } catch (error) {
        alert('Không thể truy cập camera: ' + error);
    }
}


function captureImage(videoElement) {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    canvas.getContext('2d').drawImage(videoElement, 0, 0);
    return canvas.toDataURL('image/jpeg');
}


async function captureAndRegister() {
    const idInput = document.getElementById('registerIdInput');
    const nameInput = document.getElementById('registerNameInput');
    const id = idInput.value.trim();
    const name = nameInput.value.trim();
    
    if (!id || !name) {
        alert('Vui lòng nhập đầy đủ ID và Tên');
        return;
    }

    const videoElement = document.getElementById('registerVideo');
    const imageBase64 = captureImage(videoElement);

    try {
        const response = await fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                id: id,
                name: name, 
                image: imageBase64 
            })
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            alert('Đăng ký khuôn mặt thành công!');
            closeModal('registerModal');
            idInput.value = '';
            nameInput.value = '';
        } else {
            alert('Lỗi: ' + result.message);
        }
    } catch (error) {
        alert('Lỗi kết nối: ' + error);
    }
}


async function captureAndRecognize() {
    const videoElement = document.getElementById('recognizeVideo');
    const imageBase64 = captureImage(videoElement);
    const resultsDiv = document.getElementById('results');

    try {
        const response = await fetch('/recognize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageBase64 })
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            if (result.matches.length > 0) {
                resultsDiv.innerHTML = '<h3>Kết Quả Nhận Diện:</h3>' + 
                    result.matches.map(match => 
                        `<p>ID: ${match.id}, Tên: ${match.name} (Độ tương đồng: ${(match.similarity * 100).toFixed(2)}%)</p>`
                    ).join('');
            } else {
                resultsDiv.innerHTML = '<p>Không tìm thấy khuôn mặt phù hợp</p>';
            }
            closeModal('recognizeModal');
        } else {
            resultsDiv.innerHTML = '<p>Lỗi: ' + result.message + '</p>';
        }
    } catch (error) {
        resultsDiv.innerHTML = '<p>Lỗi kết nối: ' + error + '</p>';
    }
}


function loadPersonList() {
    const select = document.getElementById('personSelect');
    select.innerHTML = '<option value="">Chọn người để xóa</option>'; 

    fetch('/list')
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            data.persons.forEach(person => {
                let option = document.createElement("option");
                option.value = person.id;
                option.textContent = person.name;
                select.appendChild(option);
            });
        } else {
            alert("Lỗi: " + data.message);
        }
    })
    .catch(error => console.error("Lỗi khi lấy danh sách:", error));
}
function openListModal() {
    const tableBody = document.getElementById('personsTableBody');
    
    fetch('/persons')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                tableBody.innerHTML = '';
                
                data.persons.forEach((person, index) => {
                    const row = tableBody.insertRow();
                    row.innerHTML = `
                        <td>${index + 1}</td>
                        <td>${person.name}</td>
                        <td>${person.image_count}</td>
                    `;
                });
            } else {
                alert('Lỗi: ' + data.message);
            }
        })
        .catch(error => {
            alert('Lỗi kết nối: ' + error);
        });
}
async function deletePerson() {
    let selectedId = document.getElementById("personSelect").value; // Đúng ID
    if (!selectedId) {
        alert("Vui lòng chọn người dùng để xóa!");
        return;
    }

    let response = await fetch("/delete", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ id: selectedId })
    });

    let result = await response.json();
    if (result.status === "success") {
        alert(result.message);
        location.reload();
    } else {
        alert("Lỗi: " + result.message);
    }
}
