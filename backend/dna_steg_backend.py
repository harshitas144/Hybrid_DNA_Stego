from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import json
import shutil
from werkzeug.utils import secure_filename
import traceback
import uuid
from datetime import datetime

# Import your DNA steganography functions
from hybrid_dna_qde_steganography import (
    embed_dna_steganography,
    extract_dna_steganography,
    calculate_dna_capacity,
    check_dna_compatibility,
    detect_file_type,
    get_file_extension
)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['TEMP_FOLDER'] = 'temp'

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['TEMP_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
ALLOWED_SECRET_EXTENSIONS = {'txt', 'pdf', 'docx', 'py', 'js', 'cpp', 'java', 'json', 'xml', 'csv', 'zip', 'rar', 'bin'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generate_session_id():
    return str(uuid.uuid4())

def cleanup_old_files():
    """Clean up files older than 1 hour"""
    current_time = datetime.now()
    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['TEMP_FOLDER']]:
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                try:
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    if (current_time - file_time).total_seconds() > 3600:  # 1 hour
                        os.remove(file_path)
                except:
                    pass

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'DNA Steganography API is running',
        'method': 'DNA encoding with Hamming error correction'
    })

@app.route('/api/check-capacity', methods=['POST'])
def check_capacity():
    """Check image capacity for DNA steganography"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': 'Invalid image file type'}), 400
        
        # Save image temporarily
        session_id = generate_session_id()
        image_filename = f"{session_id}_{secure_filename(image_file.filename)}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image_file.save(image_path)
        
        # Calculate DNA capacity
        capacity_bytes = calculate_dna_capacity(image_path)
        
        # Get image info
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                format_name = img.format
        except ImportError:
            # Fallback if PIL is not available
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                height, width = img.shape[:2]
                format_name = 'Unknown'
            else:
                width, height, format_name = 0, 0, 'Unknown'
        
        # Cleanup
        os.remove(image_path)
        
        return jsonify({
            'capacity_bytes': capacity_bytes,
            'capacity_bits': capacity_bytes * 8,
            'image_width': width,
            'image_height': height,
            'image_format': format_name,
            'method': 'DNA encoding with Hamming error correction',
            'nucleotides_capacity': width * height if width and height else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-compatibility', methods=['POST'])
def check_compatibility():
    """Check if secret file can be embedded in image using DNA steganography"""
    try:
        if 'image' not in request.files or 'secret' not in request.files:
            return jsonify({'error': 'Both image and secret files are required'}), 400
        
        image_file = request.files['image']
        secret_file = request.files['secret']
        
        if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': 'Invalid image file type'}), 400
        
        # Save files temporarily
        session_id = generate_session_id()
        image_filename = f"{session_id}_{secure_filename(image_file.filename)}"
        secret_filename = f"{session_id}_{secure_filename(secret_file.filename)}"
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        secret_path = os.path.join(app.config['UPLOAD_FOLDER'], secret_filename)
        
        image_file.save(image_path)
        secret_file.save(secret_path)
        
        # Check DNA compatibility
        is_compatible = check_dna_compatibility(image_path, secret_path)
        
        # Get file info
        secret_size = os.path.getsize(secret_path)
        secret_ext = get_file_extension(secret_filename)
        
        # Calculate capacity for detailed info
        capacity_bytes = calculate_dna_capacity(image_path)
        
        # Cleanup
        os.remove(image_path)
        os.remove(secret_path)
        
        return jsonify({
            'compatible': is_compatible,
            'secret_size': secret_size,
            'secret_extension': secret_ext,
            'capacity_bytes': capacity_bytes,
            'method': 'DNA encoding with Hamming error correction',
            'efficiency_ratio': secret_size / capacity_bytes if capacity_bytes > 0 else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/embed', methods=['POST'])
def embed_secret():
    """Embed secret file into image using DNA steganography"""
    try:
        # Validate input
        if 'image' not in request.files or 'secret' not in request.files:
            return jsonify({'error': 'Both image and secret files are required'}), 400
        
        image_file = request.files['image']
        secret_file = request.files['secret']
        password = request.form.get('password', '')
        
        if not password:
            return jsonify({'error': 'Password is required'}), 400
        
        if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': 'Invalid image file type'}), 400
        
        # Generate session ID for this operation
        session_id = generate_session_id()
        
        # Save input files
        image_filename = f"{session_id}_{secure_filename(image_file.filename)}"
        secret_filename = f"{session_id}_{secure_filename(secret_file.filename)}"
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        secret_path = os.path.join(app.config['UPLOAD_FOLDER'], secret_filename)
        
        image_file.save(image_path)
        secret_file.save(secret_path)
        
        # Check compatibility first
        if not check_dna_compatibility(image_path, secret_path):
            # Cleanup
            os.remove(image_path)
            os.remove(secret_path)
            return jsonify({'error': 'Files are not compatible for DNA embedding'}), 400
        
        # Perform DNA embedding
        output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"dna_stego_{session_id}.png")
        
        print(f"Starting DNA embedding: {image_path} -> {output_image_path}")
        
        metadata = embed_dna_steganography(
            input_image=image_path,
            secret_file=secret_path,
            output_image=output_image_path,
            password=password
        )
        
        # Get file info
        secret_size = os.path.getsize(secret_path)
        secret_ext = get_file_extension(secret_filename)
        
        # Cleanup input files
        os.remove(image_path)
        os.remove(secret_path)
        
        # Enhance metadata
        metadata['session_id'] = session_id
        metadata['secret_filename'] = secret_file.filename
        metadata['secret_size'] = secret_size
        metadata['secret_extension'] = secret_ext
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['password_protected'] = True
        
        # Save metadata
        metadata_path = os.path.join(app.config['OUTPUT_FOLDER'], f"dna_metadata_{session_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"DNA embedding completed successfully. Session: {session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'metadata': metadata,
            'message': 'Secret successfully embedded using DNA steganography',
            'output_image': f"dna_stego_{session_id}.png"
        })
        
    except Exception as e:
        print(f"DNA embedding error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/extract', methods=['POST'])
def extract_secret():
    """Extract secret file from DNA stego image"""
    stego_path = None
    try:
        print("=== DNA EXTRACTION START ===")

        # Validate input
        if 'stego_image' not in request.files:
            print("ERROR: No stego image provided")
            return jsonify({'error': 'Stego image file is required'}), 400

        stego_image_file = request.files['stego_image']
        password = request.form.get('password', '')
        metadata_json = request.form.get('metadata', '')

        print(f"Received stego image: {stego_image_file.filename}")
        print(f"Password provided: {'Yes' if password else 'No'}")
        print(f"Metadata provided: {'Yes' if metadata_json else 'No'}")

        if not password:
            print("ERROR: No password provided")
            return jsonify({'error': 'Password is required'}), 400

        if not metadata_json:
            print("ERROR: No metadata provided")
            return jsonify({'error': 'Metadata is required'}), 400

        try:
            metadata = json.loads(metadata_json)
            print(f"Metadata parsed successfully: {list(metadata.keys())}")
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid metadata format: {e}")
            return jsonify({'error': 'Invalid metadata format'}), 400

        if not allowed_file(stego_image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            print(f"ERROR: Invalid image file type: {stego_image_file.filename}")
            return jsonify({'error': 'Invalid image file type'}), 400

        # Generate session ID
        session_id = generate_session_id()
        print(f"Generated session ID: {session_id}")

        # Save stego image
        stego_filename = f"{session_id}_{secure_filename(stego_image_file.filename)}"
        stego_path = os.path.join(app.config['UPLOAD_FOLDER'], stego_filename)
        print(f"Saving stego image to: {stego_path}")
        stego_image_file.save(stego_path)

        if not os.path.exists(stego_path):
            print(f"ERROR: Failed to save stego image to {stego_path}")
            return jsonify({'error': 'Failed to save stego image'}), 500

        print(f"Stego image saved successfully, size: {os.path.getsize(stego_path)} bytes")

        # Prepare output path (without extension)
        output_base = f"dna_extracted_{session_id}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_base)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        print(f"Expected output base: {output_path}")

        print("Starting DNA extraction process...")

        # 🔁 Modified: capture returned output path
        result = extract_dna_steganography(
            stego_image=stego_path,
            metadata=metadata,
            password=password,
            output_filename=output_path
        )

        if isinstance(result, tuple):
            success, final_output_path = result
        else:
            success = result
            final_output_path = output_path if success else None

        print(f"DNA extraction completed - Success: {success}")

        if not success or not final_output_path or not os.path.exists(final_output_path):
            print("ERROR: Extracted file not found or process failed")
            return jsonify({'error': 'Failed to extract secret from image. Check password and metadata.'}), 400

        output_filename = os.path.basename(final_output_path)
        extracted_size = os.path.getsize(final_output_path)

        # Detect file type
        detected_type = 'unknown'
        try:
            with open(final_output_path, 'rb') as f:
                file_data = f.read(1024)
                detected_type = detect_file_type(file_data)
        except Exception as e:
            print(f"WARNING: Could not detect file type: {e}")

        print(f"Successfully extracted file: {output_filename}, Size: {extracted_size}")
        print("=== DNA EXTRACTION SUCCESS ===")

        return jsonify({
            'success': True,
            'session_id': session_id,
            'extracted_filename': output_filename,
            'extracted_size': extracted_size,
            'detected_type': detected_type,
            'original_filename': metadata.get('secret_filename', 'unknown'),
            'method': 'DNA steganography',
            'message': 'Secret successfully extracted using DNA steganography'
        })

    except Exception as e:
        print("=== DNA EXTRACTION FAILED ===")
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

        # Classify error
        error_msg = str(e)
        if "decrypt" in error_msg.lower():
            error_msg = "Decryption failed - check password"
        elif "metadata" in error_msg.lower():
            error_msg = "Invalid metadata - file may be corrupted"
        elif "image" in error_msg.lower():
            error_msg = "Could not process image file"
        elif "dna" in error_msg.lower():
            error_msg = "DNA sequence processing failed"

        return jsonify({
            'error': error_msg,
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

    finally:
        # Cleanup stego image
        if stego_path and os.path.exists(stego_path):
            try:
                os.remove(stego_path)
                print(f"Cleaned up stego image: {stego_path}")
            except Exception as e:
                print(f"Warning: Could not cleanup stego image: {e}")


@app.route('/api/download/<session_id>/<file_type>', methods=['GET'])
def download_file(session_id, file_type):
    """Download generated files"""
    try:
        if file_type == 'stego':
            # Download DNA stego image
            filename = f"dna_stego_{session_id}.png"
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            download_name = f"dna_stego_image_{session_id}.png"
            
        elif file_type == 'extracted':
            # Find extracted file
            extracted_files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) 
                             if f.startswith(f'dna_extracted_{session_id}')]
            if not extracted_files:
                return jsonify({'error': 'Extracted file not found'}), 404
            
            filename = extracted_files[0]
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            
            # Create a meaningful download name
            original_ext = get_file_extension(filename)
            download_name = f"dna_extracted_secret_{session_id}{original_ext}"
            
        elif file_type == 'metadata':
            # Download metadata
            filename = f"dna_metadata_{session_id}.json"
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            download_name = f"dna_metadata_{session_id}.json"
            
        else:
            return jsonify({'error': 'Invalid file type. Use: stego, extracted, or metadata'}), 400
        
        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filename}'}), 404
        
        return send_file(filepath, as_attachment=True, download_name=download_name)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup/<session_id>', methods=['DELETE'])
def cleanup_session(session_id):
    """Clean up files for a specific session"""
    try:
        cleaned_files = []
        
        # Clean up files in all folders
        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['TEMP_FOLDER']]:
            if not os.path.exists(folder):
                continue
            for filename in os.listdir(folder):
                if session_id in filename:
                    filepath = os.path.join(folder, filename)
                    try:
                        os.remove(filepath)
                        cleaned_files.append(filename)
                    except Exception as e:
                        print(f"Failed to remove {filepath}: {e}")
        
        return jsonify({
            'success': True,
            'cleaned_files': cleaned_files,
            'message': f'Cleaned up {len(cleaned_files)} files for session {session_id}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/file-info', methods=['POST'])
def get_file_info():
    """Get information about uploaded file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        session_id = generate_session_id()
        filename = f"{session_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
        file.save(filepath)
        
        # Get file info
        file_size = os.path.getsize(filepath)
        file_ext = get_file_extension(file.filename)
        
        # Detect file type from content
        detected_type = 'unknown'
        try:
            with open(filepath, 'rb') as f:
                file_data = f.read(1024)  # Read first 1KB
                detected_type = detect_file_type(file_data)
        except:
            pass
        
        # Additional info for images
        additional_info = {}
        if file_ext.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            try:
                # Try PIL first
                from PIL import Image
                with Image.open(filepath) as img:
                    additional_info['width'] = img.size[0]
                    additional_info['height'] = img.size[1]
                    additional_info['format'] = img.format
                    additional_info['mode'] = img.mode
                    # Calculate DNA capacity
                    additional_info['dna_capacity_bytes'] = calculate_dna_capacity(filepath)
            except ImportError:
                # Fallback to OpenCV
                try:
                    import cv2
                    img = cv2.imread(filepath)
                    if img is not None:
                        additional_info['height'], additional_info['width'] = img.shape[:2]
                        additional_info['dna_capacity_bytes'] = calculate_dna_capacity(filepath)
                except Exception as e:
                    additional_info['error'] = f"Could not read image: {str(e)}"
            except Exception as e:
                additional_info['error'] = f"Could not read image: {str(e)}"
        
        # Cleanup
        os.remove(filepath)
        
        return jsonify({
            'filename': file.filename,
            'size': file_size,
            'extension': file_ext,
            'detected_type': detected_type,
            'additional_info': additional_info,
            'method': 'DNA steganography analysis'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dna-stats', methods=['GET'])
def get_dna_stats():
    """Get DNA steganography statistics and information"""
    return jsonify({
        'method': 'DNA Steganography',
        'encoding': 'Binary to DNA nucleotides (A, T, G, C)',
        'mapping': {
            '00': 'A',
            '01': 'T', 
            '10': 'G',
            '11': 'C'
        },
        'error_correction': 'Hamming(7,4) code',
        'encryption': 'AES-GCM with PBKDF2',
        'security_features': [
            'Password-based encryption',
            'DNA sequence XOR encryption',
            'Hamming error correction',
            'Integrity verification',
            'Compression'
        ],
        'capacity': 'Each pixel stores 1 nucleotide (2 bits) in LSBs',
        'advantages': [
            'Bio-inspired encoding',
            'Error correction capability',
            'Additional DNA-level encryption',
            'Robust to minor image modifications'
        ]
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Periodic cleanup (run this with a scheduler in production)
@app.before_request
def before_request():
    # Clean up old files occasionally
    import random
    if random.random() < 0.1:  # 10% chance on each request
        try:
            cleanup_old_files()
        except:
            pass  # Don't let cleanup failures break the main request

if __name__ == '__main__':
    print("Starting DNA Steganography API Server...")
    print("🧬 DNA Steganography Features:")
    print("  - Binary to DNA nucleotide encoding (A, T, G, C)")
    print("  - Hamming(7,4) error correction")
    print("  - AES-GCM encryption with PBKDF2")
    print("  - DNA sequence XOR encryption")
    print("  - Automatic file type detection")
    print("  - Compression and integrity verification")
    print("")
    print("Available endpoints:")
    print("  POST /api/check-capacity - Check image DNA capacity")
    print("  POST /api/check-compatibility - Check file compatibility")
    print("  POST /api/embed - Embed secret using DNA steganography")
    print("  POST /api/extract - Extract secret from DNA stego image")
    print("  GET /api/download/<session_id>/<file_type> - Download files")
    print("  DELETE /api/cleanup/<session_id> - Clean up session files")
    print("  POST /api/file-info - Get file information")
    print("  GET /api/dna-stats - Get DNA steganography information")
    print("  GET /api/health - Health check")
    print("")
    print("🔬 DNA Encoding: 00→A, 01→T, 10→G, 11→C")
    print("🔧 Error Correction: Hamming(7,4) code")
    print("🔐 Security: AES-GCM + DNA XOR + Password")
    
    app.run(debug=True, host='0.0.0.0', port=5000)