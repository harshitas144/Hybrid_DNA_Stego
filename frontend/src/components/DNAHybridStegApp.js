import React, { useState, useRef, useCallback } from 'react';
import { Upload, Download, Lock, Unlock, Image, FileText, AlertCircle, CheckCircle, Info, Eye, EyeOff, Settings, Activity, Trash2, FileDown, Dna, Shield, Zap } from 'lucide-react';

const DNAStegApp = () => {
  const [activeTab, setActiveTab] = useState('embed');
  const [embedData, setEmbedData] = useState({
    image: null,
    secret: null,
    password: '',
    showPassword: false
  });
  const [extractData, setExtractData] = useState({
    stegoImage: null,
    password: '',
    metadata: null,
    showPassword: false
  });
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [capacityInfo, setCapacityInfo] = useState(null);
  const [compatibility, setCompatibility] = useState(null);
  const [dnaStats, setDnaStats] = useState(null);

  const imageInputRef = useRef(null);
  const secretInputRef = useRef(null);
  const stegoInputRef = useRef(null);

  const API_BASE = 'http://localhost:5000/api';

  // Load DNA stats on component mount
  React.useEffect(() => {
    fetchDnaStats();
  }, []);

  const fetchDnaStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/dna-stats`);
      if (response.ok) {
        const data = await response.json();
        setDnaStats(data);
      }
    } catch (err) {
      console.log('Could not fetch DNA stats');
    }
  };

  const handleFileUpload = useCallback((file, type, tab = 'embed') => {
    if (tab === 'embed') {
      setEmbedData(prev => ({ ...prev, [type]: file }));
    } else {
      setExtractData(prev => ({ ...prev, [type]: file }));
    }
    
    // Auto-check capacity when image is uploaded
    if (type === 'image' && tab === 'embed') {
      checkCapacity(file);
    }
    
    // Auto-check compatibility when both files are present
    if (tab === 'embed' && type === 'secret' && embedData.image) {
      checkCompatibility(embedData.image, file);
    }
  }, [embedData.image]);

  const checkCapacity = async (imageFile) => {
    if (!imageFile) return;
    
    const formData = new FormData();
    formData.append('image', imageFile);
    
    try {
      const response = await fetch(`${API_BASE}/check-capacity`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      if (response.ok) {
        setCapacityInfo(data);
      } else {
        setError(data.error || 'Failed to check capacity');
      }
    } catch (err) {
      setError('Network error while checking capacity');
    }
  };

  const checkCompatibility = async (imageFile, secretFile) => {
    if (!imageFile || !secretFile) return;
    
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('secret', secretFile);
    
    try {
      const response = await fetch(`${API_BASE}/check-compatibility`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      if (response.ok) {
        setCompatibility(data);
      } else {
        setError(data.error || 'Failed to check compatibility');
      }
    } catch (err) {
      setError('Network error while checking compatibility');
    }
  };

  const handleEmbed = async () => {
    if (!embedData.image || !embedData.secret || !embedData.password) {
      setError('Please provide image, secret file, and password');
      return;
    }
    
    setLoading(true);
    setError('');
    setSuccess('');
    
    const formData = new FormData();
    formData.append('image', embedData.image);
    formData.append('secret', embedData.secret);
    formData.append('password', embedData.password);
    
    try {
      const response = await fetch(`${API_BASE}/embed`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      if (response.ok) {
        setResults(prev => ({ ...prev, embed: data }));
        setSuccess('Secret successfully embedded using DNA steganography! You can now download the stego image and metadata.');
      } else {
        setError(data.error || 'Failed to embed secret');
      }
    } catch (err) {
      setError('Network error during embedding');
    } finally {
      setLoading(false);
    }
  };

  const handleExtract = async () => {
    if (!extractData.stegoImage || !extractData.password || !extractData.metadata) {
      setError('Please provide stego image, password, and metadata');
      return;
    }
    
    setLoading(true);
    setError('');
    setSuccess('');
    
    const formData = new FormData();
    formData.append('stego_image', extractData.stegoImage);
    formData.append('password', extractData.password);
    formData.append('metadata', JSON.stringify(extractData.metadata));
    
    try {
      const response = await fetch(`${API_BASE}/extract`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      if (response.ok) {
        setResults(prev => ({ ...prev, extract: data }));
        setSuccess('Secret successfully extracted using DNA steganography! You can now download the recovered file.');
      } else {
        setError(data.error || 'Failed to extract secret');
      }
    } catch (err) {
      setError('Network error during extraction');
    } finally {
      setLoading(false);
    }
  };

  const downloadFile = async (sessionId, fileType) => {
    try {
      const response = await fetch(`${API_BASE}/download/${sessionId}/${fileType}`);
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${fileType}_${sessionId}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      } else {
        const error = await response.json();
        setError(error.error || 'Failed to download file');
      }
    } catch (err) {
      setError('Network error during download');
    }
  };

  const cleanupSession = async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE}/cleanup/${sessionId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        setSuccess('Session files cleaned up successfully');
      }
    } catch (err) {
      console.log('Could not cleanup session');
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const FileUploadArea = ({ onFileSelect, accept, label, icon: Icon, file }) => (
    <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
      <div className="flex flex-col items-center space-y-4">
        <Icon className="w-12 h-12 text-gray-400" />
        <div>
          <p className="text-lg font-medium text-gray-700">{label}</p>
          {file && (
            <p className="text-sm text-green-600 mt-2">
              Selected: {file.name} ({formatBytes(file.size)})
            </p>
          )}
        </div>
        <input
          type="file"
          accept={accept}
          onChange={(e) => onFileSelect(e.target.files[0])}
          className="hidden"
          id={`file-${label.replace(/\s+/g, '-').toLowerCase()}`}
        />
        <label
          htmlFor={`file-${label.replace(/\s+/g, '-').toLowerCase()}`}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 cursor-pointer transition-colors"
        >
          Choose File
        </label>
      </div>
    </div>
  );

  const InfoPanel = ({ title, data, variant = 'info' }) => {
    const bgColor = variant === 'success' ? 'bg-green-50 border-green-200' : 
                   variant === 'warning' ? 'bg-yellow-50 border-yellow-200' : 
                   'bg-blue-50 border-blue-200';
    
    return (
      <div className={`rounded-lg p-4 border ${bgColor}`}>
        <h3 className="font-semibold text-gray-800 mb-2">{title}</h3>
        <div className="space-y-1 text-sm">
          {Object.entries(data).map(([key, value]) => (
            <div key={key} className="flex justify-between">
              <span className="text-gray-600">{key}:</span>
              <span className="font-medium">{value}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const DNAVisualization = () => (
    <div className="bg-gradient-to-r from-purple-100 to-blue-100 rounded-lg p-4 mb-6">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-800 flex items-center">
          <Dna className="w-5 h-5 mr-2 text-purple-600" />
          DNA Encoding Method
        </h3>
        <div className="flex items-center space-x-2">
          <Shield className="w-4 h-4 text-green-600" />
          <span className="text-sm text-green-600">Hamming Error Correction</span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
        <div className="text-center p-2 bg-white rounded border">
          <div className="font-mono text-lg font-bold text-purple-600">00 → A</div>
          <div className="text-xs text-gray-600">Adenine</div>
        </div>
        <div className="text-center p-2 bg-white rounded border">
          <div className="font-mono text-lg font-bold text-blue-600">01 → T</div>
          <div className="text-xs text-gray-600">Thymine</div>
        </div>
        <div className="text-center p-2 bg-white rounded border">
          <div className="font-mono text-lg font-bold text-green-600">10 → G</div>
          <div className="text-xs text-gray-600">Guanine</div>
        </div>
        <div className="text-center p-2 bg-white rounded border">
          <div className="font-mono text-lg font-bold text-red-600">11 → C</div>
          <div className="text-xs text-gray-600">Cytosine</div>
        </div>
      </div>
      
      <p className="text-sm text-gray-600 text-center">
        Each pixel stores 1 nucleotide (2 bits) with AES-GCM encryption + DNA XOR protection
      </p>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center mb-4">
              <Dna className="w-12 h-12 text-purple-600 mr-3" />
              <h1 className="text-4xl font-bold text-gray-800">
                DNA Steganography
              </h1>
            </div>
            <p className="text-gray-600 mb-2">
              Bio-inspired secure data hiding using DNA nucleotide encoding
            </p>
            <div className="flex items-center justify-center space-x-4 text-sm text-gray-500">
              <div className="flex items-center">
                <Shield className="w-4 h-4 mr-1" />
                <span>Hamming(7,4) Error Correction</span>
              </div>
              <div className="flex items-center">
                <Zap className="w-4 h-4 mr-1" />
                <span>AES-GCM Encryption</span>
              </div>
            </div>
          </div>

          {/* DNA Visualization */}
          <DNAVisualization />

          {/* Tab Navigation */}
          <div className="flex space-x-1 mb-6">
            <button
              onClick={() => setActiveTab('embed')}
              className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors ${
                activeTab === 'embed'
                  ? 'bg-purple-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              <Lock className="inline w-4 h-4 mr-2" />
              Embed Secret
            </button>
            <button
              onClick={() => setActiveTab('extract')}
              className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors ${
                activeTab === 'extract'
                  ? 'bg-purple-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              <Unlock className="inline w-4 h-4 mr-2" />
              Extract Secret
            </button>
          </div>

          {/* Alert Messages */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center">
              <AlertCircle className="w-5 h-5 text-red-500 mr-3" />
              <span className="text-red-700">{error}</span>
              <button
                onClick={() => setError('')}
                className="ml-auto text-red-500 hover:text-red-700"
              >
                ×
              </button>
            </div>
          )}

          {success && (
            <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg flex items-center">
              <CheckCircle className="w-5 h-5 text-green-500 mr-3" />
              <span className="text-green-700">{success}</span>
              <button
                onClick={() => setSuccess('')}
                className="ml-auto text-green-500 hover:text-green-700"
              >
                ×
              </button>
            </div>
          )}

          {/* Main Content */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            {activeTab === 'embed' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
                  <Dna className="w-6 h-6 mr-2 text-purple-600" />
                  Embed Secret using DNA Encoding
                </h2>
                
                {/* File Upload Areas */}
                <div className="grid md:grid-cols-2 gap-6">
                  <FileUploadArea
                    onFileSelect={(file) => handleFileUpload(file, 'image', 'embed')}
                    accept="image/*"
                    label="Cover Image"
                    icon={Image}
                    file={embedData.image}
                  />
                  <FileUploadArea
                    onFileSelect={(file) => handleFileUpload(file, 'secret', 'embed')}
                    accept="*"
                    label="Secret File"
                    icon={FileText}
                    file={embedData.secret}
                  />
                </div>

                {/* Password Input */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
                    <Shield className="w-5 h-5 mr-2" />
                    Encryption Settings
                  </h3>
                  
                  <div className="relative">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Password (Required for AES-GCM + DNA XOR encryption)
                    </label>
                    <div className="relative">
                      <input
                        type={embedData.showPassword ? 'text' : 'password'}
                        value={embedData.password}
                        onChange={(e) => setEmbedData(prev => ({ ...prev, password: e.target.value }))}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent pr-10"
                        placeholder="Enter strong encryption password"
                      />
                      <button
                        type="button"
                        onClick={() => setEmbedData(prev => ({ ...prev, showPassword: !prev.showPassword }))}
                        className="absolute right-3 top-2.5 text-gray-500 hover:text-gray-700"
                      >
                        {embedData.showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Capacity Information */}
                {capacityInfo && (
                  <InfoPanel
                    title="DNA Capacity Analysis"
                    data={{
                      'DNA Capacity': `${formatBytes(capacityInfo.capacity_bytes)} (${capacityInfo.capacity_bits} bits)`,
                      'Image Size': `${capacityInfo.image_width} × ${capacityInfo.image_height}`,
                      'Format': capacityInfo.image_format,
                      'Nucleotides': capacityInfo.nucleotides_capacity?.toLocaleString() || 'N/A',
                      'Method': capacityInfo.method
                    }}
                  />
                )}

                {/* Compatibility Check */}
                {compatibility && (
                  <InfoPanel
                    title="DNA Compatibility Check"
                    data={{
                      'Status': compatibility.compatible ? '✅ Compatible' : '❌ Not Compatible',
                      'Secret Size': formatBytes(compatibility.secret_size),
                      'File Type': compatibility.secret_extension,
                      'Efficiency': `${(compatibility.efficiency_ratio * 100).toFixed(1)}%`,
                      'Method': compatibility.method
                    }}
                    variant={compatibility.compatible ? 'success' : 'warning'}
                  />
                )}

                {/* Embed Button */}
                <button
                  onClick={handleEmbed}
                  disabled={loading || !embedData.image || !embedData.secret || !embedData.password}
                  className="w-full bg-purple-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {loading ? (
                    <>
                      <Activity className="w-5 h-5 animate-spin" />
                      <span>Encoding to DNA...</span>
                    </>
                  ) : (
                    <>
                      <Dna className="w-5 h-5" />
                      <span>Embed using DNA Encoding</span>
                    </>
                  )}
                </button>

                {/* Embed Results */}
                {results.embed && (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <h3 className="font-semibold text-green-800 mb-3">DNA Embedding Complete!</h3>
                    <div className="space-y-3">
                      <InfoPanel
                        title="DNA Embedding Metadata"
                        data={{
                          'Method': results.embed.metadata.method || 'DNA Steganography',
                          'Secret File': results.embed.metadata.secret_filename,
                          'Secret Size': formatBytes(results.embed.metadata.secret_size),
                          'File Type': results.embed.metadata.secret_extension,
                          'Timestamp': new Date(results.embed.metadata.timestamp).toLocaleString(),
                          'Protected': results.embed.metadata.password_protected ? '🔒 Yes' : '🔓 No'
                        }}
                        variant="success"
                      />
                      <div className="flex space-x-3">
                        <button
                          onClick={() => downloadFile(results.embed.session_id, 'stego')}
                          className="flex-1 bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 flex items-center justify-center space-x-2"
                        >
                          <Download className="w-4 h-4" />
                          <span>Download Stego Image</span>
                        </button>
                        <button
                          onClick={() => downloadFile(results.embed.session_id, 'metadata')}
                          className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 flex items-center justify-center space-x-2"
                        >
                          <FileDown className="w-4 h-4" />
                          <span>Download Metadata</span>
                        </button>
                        <button
                          onClick={() => cleanupSession(results.embed.session_id)}
                          className="px-4 py-2 text-gray-600 hover:text-red-600 border border-gray-300 rounded-lg hover:border-red-300"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'extract' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
                  <Dna className="w-6 h-6 mr-2 text-purple-600" />
                  Extract Secret from DNA Encoding
                </h2>
                
                {/* File Upload */}
                <FileUploadArea
                  onFileSelect={(file) => handleFileUpload(file, 'stegoImage', 'extract')}
                  accept="image/*"
                  label="DNA Stego Image"
                  icon={Image}
                  file={extractData.stegoImage}
                />

                {/* Metadata Upload */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
                    <Info className="w-5 h-5 mr-2" />
                    DNA Metadata
                  </h3>
                  <div className="space-y-3">
                    <input
                      type="file"
                      accept=".json"
                      onChange={(e) => {
                        const file = e.target.files[0];
                        if (file) {
                          const reader = new FileReader();
                          reader.onload = (e) => {
                            try {
                              const metadata = JSON.parse(e.target.result);
                              setExtractData(prev => ({ ...prev, metadata }));
                            } catch (err) {
                              setError('Invalid metadata JSON file');
                            }
                          };
                          reader.readAsText(file);
                        }
                      }}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    />
                    <p className="text-sm text-gray-600">
                      Upload the DNA metadata JSON file generated during embedding
                    </p>
                    {extractData.metadata && (
                      <div className="text-sm text-green-600">
                        ✅ Metadata loaded: {extractData.metadata.secret_filename || 'Unknown file'}
                      </div>
                    )}
                  </div>
                </div>

                {/* Password */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
                    <Shield className="w-5 h-5 mr-2" />
                    Decryption Password
                  </h3>
                  <div className="relative">
                    <input
                      type={extractData.showPassword ? 'text' : 'password'}
                      value={extractData.password}
                      onChange={(e) => setExtractData(prev => ({ ...prev, password: e.target.value }))}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent pr-10"
                      placeholder="Enter decryption password"
                    />
                    <button
                      type="button"
                      onClick={() => setExtractData(prev => ({ ...prev, showPassword: !prev.showPassword }))}
                      className="absolute right-3 top-2.5 text-gray-500 hover:text-gray-700"
                    >
                      {extractData.showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                </div>

                {/* Extract Button */}
                <button
                  onClick={handleExtract}
                  disabled={loading || !extractData.stegoImage || !extractData.password || !extractData.metadata}
                  className="w-full bg-purple-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {loading ? (
                    <>
                      <Activity className="w-5 h-5 animate-spin" />
                      <span>Decoding DNA...</span>
                    </>
                  ) : (
                    <>
                      <Dna className="w-5 h-5" />
                      <span>Extract from DNA Encoding</span>
                    </>
                  )}
                </button>

                {/* Extract Results */}
                {results.extract && (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <h3 className="font-semibold text-green-800 mb-3">DNA Extraction Complete!</h3>
                    <div className="space-y-3">
                      <InfoPanel
                        title="Extracted File Information"
                        data={{
                          'Original Filename': results.extract.original_filename,
                          'Extracted File': results.extract.extracted_filename,
                          'File Size': formatBytes(results.extract.extracted_size),
                          'Detected Type': results.extract.detected_type,
                          'Method': results.extract.method,
                          'Status': '✅ Successfully decoded from DNA'
                        }}
                        variant="success"
                      />
                      <div className="flex space-x-3">
                        <button
                          onClick={() => downloadFile(results.extract.session_id, 'extracted')}
                          className="flex-1 bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 flex items-center justify-center space-x-2"
                        >
                          <Download className="w-4 h-4" />
                          <span>Download Extracted File</span>
                        </button>
                        <button
                          onClick={() => cleanupSession(results.extract.session_id)}
                          className="px-4 py-2 text-gray-600 hover:text-red-600 border border-gray-300 rounded-lg hover:border-red-300"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
};

export default DNAStegApp;
