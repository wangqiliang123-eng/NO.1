from flask import Flask, request, jsonify
from subtitle_extractor import extract_subtitle
import os

app = Flask(__name__)

@app.route('/api/extract', methods=['POST'])
def extract():
    try:
        video_path = request.json['video_path']
        output_path = request.json.get('output_path', 'output/subtitles.srt')
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 调用现有的字幕提取函数
        result = extract_subtitle(video_path, output_path)
        
        return jsonify({
            'success': True,
            'output_path': output_path,
            'message': '字幕提取完成'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=5000) 