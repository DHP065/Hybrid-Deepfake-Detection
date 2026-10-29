import streamlit as st
import os
import tempfile
import json
import time
from hybrid_deepfake_detector import HybridDeepfakeDetector
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Hybrid Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .fake-result {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border: 3px solid #f44336;
        color: #c62828;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        animation: pulse-red 2s infinite;
    }
    .real-result {
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
        border: 3px solid #4caf50;
        color: #2e7d32;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3); }
        50% { box-shadow: 0 6px 20px rgba(244, 67, 54, 0.5); }
    }
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3); }
        50% { box-shadow: 0 6px 20px rgba(76, 175, 80, 0.5); }
    }
    .detection-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        margin: 5px;
    }
    .badge-fake {
        background-color: #f44336;
        color: white;
        border: 2px solid #d32f2f;
    }
    .badge-real {
        background-color: #4caf50;
        color: white;
        border: 2px solid #388e3c;
    }
    .confidence-bar {
        width: 100%;
        height: 20px;
        background-color: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-fill-fake {
        height: 100%;
        background: linear-gradient(90deg, #ffeb3b, #ff9800, #f44336);
        transition: width 0.8s ease-in-out;
    }
    .confidence-fill-real {
        height: 100%;
        background: linear-gradient(90deg, #ffeb3b, #8bc34a, #4caf50);
        transition: width 0.8s ease-in-out;
    }
    .metric-card {
        background-color: black;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None

def load_detector():
    """Load the deepfake detector with caching"""
    if st.session_state.detector is None:
        with st.spinner("Loading AI model..."):
            st.session_state.detector = HybridDeepfakeDetector()
    return st.session_state.detector

def create_metrics_visualization(results):
    """Create interactive visualizations for the analysis results"""
    
    # Create metrics dataframe
    metrics_data = {
        'Metric': ['CNN Confidence', 'rPPG SNR', 'Noise Score', 'Final Score'],
        'Value': [
            results.get('pretrained_prob_fake', 0) * 100,
            min(results.get('rppg_snr', 0), 100),  # Cap at 100 for visualization
            results.get('noise_score', 0) * 100,
            results.get('final_score', 0) * 100
        ],
        'Description': [
            'Deep learning model confidence (%)',
            'Heart signal quality (higher = better)',
            'Video noise/artifacts (%)',
            'Combined analysis score (%)'
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Create bar chart
    fig = px.bar(
        df, 
        x='Metric', 
        y='Value',
        title='Detection Analysis Breakdown',
        color='Value',
        color_continuous_scale='RdYlBu_r',
        hover_data=['Description']
    )
    
    fig.update_layout(
        xaxis_title="Analysis Components",
        yaxis_title="Score/Confidence",
        showlegend=False,
        height=400
    )
    
    return fig

def create_confidence_gauge(final_score):
    """Create a gauge chart for the final confidence score"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = final_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Deepfake Confidence"},
        delta = {'reference': 30},  # Threshold for fake detection
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def generate_pdf_report(results, video_filename="uploaded_video"):
    """Generate a comprehensive PDF report of the deepfake analysis"""
    
    # Create a BytesIO buffer to store the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # Center alignment
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    # Title
    story.append(Paragraph("🔍 Hybrid Deepfake Detection Report", title_style))
    story.append(Spacer(1, 20))
    
    # Report metadata
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Analysis Date:</b> {current_time}", styles['Normal']))
    story.append(Paragraph(f"<b>Video File:</b> {video_filename}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Main Result
    decision = results.get('decision', 'UNKNOWN')
    final_score = results.get('final_score', 0)
    
    if decision == 'FAKE':
        result_color = colors.red
        result_text = f"🚨 DEEPFAKE DETECTED (Confidence: {final_score*100:.1f}%)"
    else:
        result_color = colors.green
        result_text = f"✅ APPEARS GENUINE (Confidence: {(1-final_score)*100:.1f}%)"
    
    result_style = ParagraphStyle(
        'ResultStyle',
        parent=styles['Normal'],
        fontSize=18,
        alignment=1,
        textColor=result_color,
        borderWidth=2,
        borderColor=result_color,
        borderPadding=10,
        spaceAfter=20
    )
    
    story.append(Paragraph(result_text, result_style))
    story.append(Spacer(1, 20))
    
    # Analysis Summary Table
    story.append(Paragraph("📊 Analysis Summary", heading_style))
    
    summary_data = [
        ['Metric', 'Value', 'Description'],
        ['Final Score', f"{final_score*100:.1f}%", 'Overall deepfake probability'],
        ['CNN Confidence', f"{results.get('pretrained_prob_fake', 0)*100:.1f}%", 'Deep learning model prediction'],
        ['rPPG SNR', f"{results.get('rppg_snr', 0):.1f}", 'Heart signal quality'],
        ['Peak BPM', f"{results.get('rppg_peak_bpm', 0):.1f}", 'Detected heart rate'],
        ['Noise Score', f"{results.get('noise_score', 0)*1000:.2f}", 'Video artifacts level']
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Detailed Analysis
    story.append(Paragraph("🔬 Detailed Analysis", heading_style))
    
    # CNN Analysis
    story.append(Paragraph("<b>🧠 Deep Learning Analysis:</b>", styles['Heading3']))
    cnn_prob = results.get('pretrained_prob_fake', 0)
    if cnn_prob > 0.7:
        cnn_assessment = "High probability of manipulation detected"
    elif cnn_prob > 0.3:
        cnn_assessment = "Moderate signs of manipulation"
    else:
        cnn_assessment = "Low probability of manipulation"
    
    story.append(Paragraph(f"The ResNet18 model trained on FaceForensics++ dataset indicates a {cnn_prob*100:.1f}% probability that this video contains deepfake content. {cnn_assessment}.", styles['Normal']))
    story.append(Spacer(1, 10))
    
    # rPPG Analysis
    story.append(Paragraph("<b>💓 Heart Signal Analysis (rPPG):</b>", styles['Heading3']))
    rppg_snr = results.get('rppg_snr', 0)
    rppg_bpm = results.get('rppg_peak_bpm', 0)
    
    if rppg_snr is not None and rppg_snr > 0:
        if rppg_snr < 5:
            rppg_assessment = "Poor heart signal quality suggests potential manipulation"
        elif rppg_snr < 20:
            rppg_assessment = "Moderate heart signal quality"
        else:
            rppg_assessment = "Good heart signal quality detected"
        
        story.append(Paragraph(f"Signal-to-Noise Ratio: {rppg_snr:.1f}. Peak heart rate detected at {rppg_bpm:.1f} BPM. {rppg_assessment}.", styles['Normal']))
    else:
        story.append(Paragraph("Heart signal analysis could not be performed due to insufficient facial detection.", styles['Normal']))
    
    story.append(Spacer(1, 10))
    
    # Noise Analysis
    story.append(Paragraph("<b>🔊 Video Noise Analysis:</b>", styles['Heading3']))
    noise_score = results.get('noise_score', 0)
    if noise_score > 0.001:
        noise_assessment = "Higher noise levels detected, possibly indicating compression artifacts from deepfake generation"
    else:
        noise_assessment = "Normal noise levels detected"
    
    story.append(Paragraph(f"Noise score: {noise_score*1000:.3f}. {noise_assessment}.", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Technical Notes
    story.append(Paragraph("⚙️ Technical Notes", heading_style))
    notes = results.get('notes', [])
    for i, note in enumerate(notes, 1):
        story.append(Paragraph(f"{i}. {note}", styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Methodology
    story.append(Paragraph("🛠️ Detection Methodology", heading_style))
    methodology_text = """
    This analysis employs a hybrid approach combining multiple detection techniques:
    
    • <b>Convolutional Neural Network (CNN):</b> A ResNet18 model trained on the FaceForensics++ dataset 
      to detect visual artifacts typical of deepfake generation.
    
    • <b>Remote Photoplethysmography (rPPG):</b> Analysis of subtle color changes in facial regions 
      to detect authentic heart signals, which are difficult to replicate in deepfakes.
    
    • <b>Noise and Compression Analysis:</b> Detection of artifacts introduced during the deepfake 
      generation process, including unusual compression patterns and noise characteristics.
    
    The final decision is based on a weighted combination of all three methods, providing 
    robust and reliable deepfake detection.
    """
    
    story.append(Paragraph(methodology_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=10,
        alignment=1,
        textColor=colors.grey
    )
    
    story.append(Paragraph("Generated by Hybrid Deepfake Detector v1.0", footer_style))
    
    # Build PDF
    doc.build(story)
    
    # Get the PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def main():
    # Header
    st.markdown('<div class="main-header">🔍 Hybrid Deepfake Detector</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;">
    Advanced AI system that combines multiple detection techniques:<br>
    🧠 <strong>Deep Learning</strong> • 💓 <strong>Heart Signal Analysis</strong> • 🔊 <strong>Noise Detection</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 How it works")
        st.markdown("""
        1. **Upload** a video file
        2. **AI Analysis** using multiple techniques:
           - CNN model trained on deepfakes
           - rPPG heart signal detection
           - Video noise/artifact analysis
        3. **Get Results** with detailed breakdown
        """)
        
        st.header("📊 Supported Formats")
        st.markdown("• MP4, AVI, MOV, MKV")
        st.markdown("• Max file size: 200MB")
        
        st.header("⚡ Model Info")
        if st.session_state.detector:
            st.success("✅ Model loaded and ready")
            st.info("🎯 ResNet18 + Hybrid Detection")
        else:
            st.info("🔄 Model will load on first use")
        
        st.header("🎯 Detection Features")
        st.markdown("""
        • **REAL/FAKE Classification**
        • **Confidence Percentage**
        • **Threat Level Assessment**
        • **Reliability Scoring**
        • **Visual Result Indicators**
        • **Detailed PDF Reports**
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Video")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze for deepfake detection"
        )
        
        # Store uploaded file info in session state
        if uploaded_file is not None:
            st.session_state.uploaded_filename = uploaded_file.name
        
        if uploaded_file is not None:
            # Display video info
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {uploaded_file.size / (1024*1024):.1f} MB")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name
            
            # Show video preview
            st.video(uploaded_file)
            
            # Analysis button
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                # Load detector
                detector = load_detector()
                
                # Run analysis with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Initializing analysis...")
                    progress_bar.progress(10)
                    
                    status_text.text("Extracting video frames...")
                    progress_bar.progress(30)
                    
                    status_text.text("Running AI analysis...")
                    progress_bar.progress(60)
                    
                    # Run the actual analysis
                    results = detector.analyze(temp_video_path, out_dir="temp_out")
                    
                    status_text.text("Finalizing results...")
                    progress_bar.progress(90)
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.analysis_complete = True
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Clean up temp file
                    os.unlink(temp_video_path)
                    
                    time.sleep(0.5)  # Brief pause for user experience
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)
    
    with col2:
        st.header("📊 Analysis Results")
        
        if st.session_state.analysis_complete and st.session_state.results:
            results = st.session_state.results
            
            # Main result with enhanced display
            decision = results.get('decision', 'UNKNOWN')
            final_score = results.get('final_score', 0)
            
            # Do not override decision in UI; use model's fusion result
            
            # Create columns for better layout
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            
            with result_col2:
                if decision == 'FAKE':
                    confidence = final_score * 100
                    st.markdown(f"""
                    <div class="result-box fake-result">
                        <h1 style="margin: 0; font-size: 2.5rem;">🚨 DEEPFAKE DETECTED</h1>
                        <h2 style="margin: 10px 0; color: #c62828;">⚠️ This video appears to be FAKE</h2>
                        <div style="font-size: 1.5rem; margin: 15px 0;">
                            Confidence: <strong>{confidence:.1f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add confidence bar
                    st.markdown(f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <p><strong>Detection Confidence Level:</strong></p>
                        <div class="confidence-bar">
                            <div class="confidence-fill-fake" style="width: {confidence}%;"></div>
                        </div>
                        <p style="font-size: 0.9rem; color: #666;">
                            {confidence:.1f}% confidence this video contains deepfake content
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    confidence = (1 - final_score) * 100
                    st.markdown(f"""
                    <div class="result-box real-result">
                        <h1 style="margin: 0; font-size: 2.5rem;">✅ APPEARS GENUINE</h1>
                        <h2 style="margin: 10px 0; color: #2e7d32;">✓ This video appears to be REAL</h2>
                        <div style="font-size: 1.5rem; margin: 15px 0;">
                            Confidence: <strong>{confidence:.1f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add confidence bar
                    st.markdown(f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <p><strong>Authenticity Confidence Level:</strong></p>
                        <div class="confidence-bar">
                            <div class="confidence-fill-real" style="width: {confidence}%;"></div>
                        </div>
                        <p style="font-size: 0.9rem; color: #666;">
                            {confidence:.1f}% confidence this video is authentic
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add summary badges
            st.markdown("---")
            
            col_badge1, col_badge2, col_badge3 = st.columns(3)
            
            with col_badge1:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div class="detection-badge {'badge-fake' if decision == 'FAKE' else 'badge-real'}">
                        STATUS: {decision}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_badge2:
                threat_level = "HIGH RISK" if final_score > 0.7 else "MEDIUM RISK" if final_score > 0.3 else "LOW RISK"
                threat_color = "#f44336" if final_score > 0.7 else "#ff9800" if final_score > 0.3 else "#4caf50"
                st.markdown(f"""
                <div style="text-align: center;">
                    <div class="detection-badge" style="background-color: {threat_color}; color: white; border: 2px solid {threat_color};">
                        THREAT: {threat_level}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_badge3:
                reliability = "HIGH" if abs(final_score - 0.5) > 0.3 else "MEDIUM" if abs(final_score - 0.5) > 0.1 else "LOW"
                reliability_color = "#4caf50" if reliability == "HIGH" else "#ff9800" if reliability == "MEDIUM" else "#f44336"
                st.markdown(f"""
                <div style="text-align: center;">
                    <div class="detection-badge" style="background-color: {reliability_color}; color: white; border: 2px solid {reliability_color};">
                        RELIABILITY: {reliability}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add interpretation summary
            st.markdown("---")
            st.subheader("📋 Quick Summary")
            
            if decision == 'FAKE':
                interpretation = "⚠️ **WARNING**: This video has been identified as a deepfake with high confidence. "
                if final_score > 0.7:
                    interpretation += "The AI detected strong indicators of manipulation. This video should NOT be trusted as authentic content."
                elif final_score > 0.5:
                    interpretation += "The AI detected moderate indicators of manipulation. Exercise caution when sharing or believing this content."
                else:
                    interpretation += "The AI detected some potential signs of manipulation, but the confidence is lower."
            else:
                interpretation = "✅ **GOOD NEWS**: This video appears to be authentic. "
                real_confidence = (1 - final_score) * 100
                if real_confidence > 70:
                    interpretation += "The AI found strong evidence that this is genuine, unmanipulated content."
                elif real_confidence > 50:
                    interpretation += "The AI believes this is likely genuine content, though some uncertainty remains."
                else:
                    interpretation += "The AI leans towards this being real, but with lower confidence."
            
            st.markdown(f"""
            <div style="background-color: {'#fff3cd' if decision == 'FAKE' else '#d4edda'}; 
                        border: 1px solid {'#ffeaa7' if decision == 'FAKE' else '#c3e6cb'}; 
                        border-radius: 8px; padding: 15px; margin: 10px 0;">
                <p style="margin: 0; color: {'#856404' if decision == 'FAKE' else '#155724'}; font-size: 1.1rem;">
                    {interpretation}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge
            st.plotly_chart(create_confidence_gauge(final_score), use_container_width=True)
            
            # Detailed metrics
            st.subheader("📈 Detailed Analysis")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>🧠 CNN Analysis</strong><br>
                    Fake Probability: {results.get('pretrained_prob_fake', 0)*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <strong>💓 Heart Signal (rPPG)</strong><br>
                    SNR: {results.get('rppg_snr', 0):.1f}<br>
                    Peak BPM: {results.get('rppg_peak_bpm', 0):.1f}
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>🔊 Noise Analysis</strong><br>
                    Noise Score: {results.get('noise_score', 0)*1000:.2f}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <strong>⚖️ Final Score</strong><br>
                    Combined: {final_score*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization
            st.plotly_chart(create_metrics_visualization(results), use_container_width=True)
            
            # Technical notes
            with st.expander("🔬 Technical Details"):
                notes = results.get('notes', [])
                for note in notes:
                    st.write(f"• {note}")
                
                st.write("**Detection Method**: Hybrid analysis combining:")
                st.write("- ResNet18 CNN trained on FaceForensics++")
                st.write("- Remote photoplethysmography (rPPG) for heart signal")
                st.write("- Video compression artifacts and noise analysis")
            
            # Download results
            if st.button("📥 Download Detailed Report", use_container_width=True):
                # Get the original filename from session state
                video_name = st.session_state.get('uploaded_filename', 'analysis_video')
                
                pdf_data = generate_pdf_report(results, video_name)
                st.download_button(
                    label="💾 Save Report as PDF",
                    data=pdf_data,
                    file_name=f"deepfake_report_{int(time.time())}.pdf",
                    mime="application/pdf"
                )
        
        else:
            st.info("👆 Upload a video file and click 'Analyze Video' to see results")
            
            # Add helpful information
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3 style="color: #333; margin-top: 0;">🔍 How Detection Works</h3>
                <p>Our AI system analyzes videos using multiple techniques:</p>
                <ul style="color: #666;">
                    <li><strong>🧠 Deep Learning:</strong> CNN model trained on thousands of deepfake videos</li>
                    <li><strong>💓 Heart Signal Analysis:</strong> Detects authentic biological signals in facial regions</li>
                    <li><strong>🔊 Noise Analysis:</strong> Identifies compression artifacts from deepfake generation</li>
                </ul>
                <p style="margin-bottom: 0;"><strong>Results are displayed with clear REAL/FAKE indicators and confidence levels.</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show example with sample video
            if st.button("🎬 Try with Sample Video", use_container_width=True):
                if os.path.exists("samples/fake.mp4"):
                    detector = load_detector()
                    
                    with st.spinner("Analyzing sample video..."):
                        results = detector.analyze("samples/fake.mp4", out_dir="temp_out")
                        st.session_state.results = results
                        st.session_state.analysis_complete = True
                        st.session_state.uploaded_filename = "fake.mp4"  # Set filename for PDF generation
                        st.rerun()
                else:
                    st.error("Sample video not found!")

if __name__ == "__main__":
    main()
