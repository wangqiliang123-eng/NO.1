document.getElementById('extractBtn').addEventListener('click', async () => {
    const videoInput = document.getElementById('videoInput');
    const file = videoInput.files[0];
    if (!file) {
        alert('请选择视频文件');
        return;
    }

    const status = document.getElementById('status');
    status.textContent = '正在提取字幕...';

    try {
        const response = await fetch('http://localhost:5000/api/extract', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                video_path: file.path,
                output_path: 'output/subtitles.srt'
            })
        });

        const result = await response.json();
        if (result.success) {
            status.textContent = '字幕提取完成！';
        } else {
            status.textContent = `错误：${result.error}`;
        }
    } catch (error) {
        status.textContent = `错误：${error.message}`;
    }
}); 