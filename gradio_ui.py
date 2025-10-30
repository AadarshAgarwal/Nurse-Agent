#!/usr/bin/env python3
"""
Gradio Web UI for Fever Proforma Nurse Agent
===========================================

Modern web interface with recording button for the fever assessment system.
"""

import gradio as gr
import json
import threading
import time
from datetime import datetime
from live_nurse_agent import FeverProformaNurseAgent

class NurseAgentUI:
    def __init__(self):
        """Initialize the Nurse Agent UI"""
        try:
            self.nurse_agent = FeverProformaNurseAgent()
            self.initialized = True
            self.is_recording = False
            self.recording_thread = None
            print("✅ Nurse Agent initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Nurse Agent: {e}")
            self.initialized = False
            self.is_recording = False
    
    def start_recording(self):
        """
        Start recording audio using Whisper
        
        Returns:
            tuple: (status_message, start_button_state, stop_button_state)
        """
        if not self.initialized:
            return "❌ System not initialized properly", gr.Button(interactive=True), gr.Button(interactive=False)
        
        if self.is_recording:
            return "⚠️ Already recording! Click stop to end recording.", gr.Button(interactive=False), gr.Button(interactive=True)
        
        try:
            # Start Whisper recording
            success, message = self.nurse_agent.start_whisper_recording()
            
            if success:
                self.is_recording = True
                
                # Start continuous recording in background thread
                def continuous_record():
                    while self.is_recording:
                        if not self.nurse_agent.record_audio_continuously():
                            break
                        time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
                self.recording_thread = threading.Thread(target=continuous_record)
                self.recording_thread.daemon = True
                self.recording_thread.start()
                
                return "🎤 Recording started with Whisper! Speak now... Click STOP when finished.", gr.Button(interactive=False), gr.Button(interactive=True)
            else:
                return f"❌ Failed to start recording: {message}", gr.Button(interactive=True), gr.Button(interactive=False)
            
        except Exception as e:
            return f"❌ Error starting recording: {str(e)}", gr.Button(interactive=True), gr.Button(interactive=False)
    
    def stop_recording(self):
        """
        Stop recording and process audio with Whisper
        
        Returns:
            tuple: (status_message, transcribed_text, patient_data, start_button_state, stop_button_state)
        """
        if not self.is_recording:
            return "⚠️ Not currently recording", "", "", gr.Button(interactive=True), gr.Button(interactive=False)
        
        try:
            # Stop recording
            self.is_recording = False
            
            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2)
            
            # Get transcribed text using Whisper
            transcribed_text = self.nurse_agent.stop_whisper_recording()
            
            if transcribed_text.startswith("Error") or transcribed_text.startswith("Could not") or transcribed_text.startswith("No"):
                return f"❌ Recording failed: {transcribed_text}", "", "", gr.Button(interactive=True), gr.Button(interactive=False)
            
            # Extract patient data from speech
            patient_data = self.nurse_agent.extract_patient_data_from_speech(transcribed_text)
            
            # Format patient data for display
            formatted_data = self.nurse_agent.format_patient_data_for_llm(patient_data)
            
            status_msg = f"✅ Recording stopped and processed with Whisper!\n📊 Captured {len(transcribed_text)} characters of speech\n🔄 Patient data extracted successfully"
            
            return status_msg, transcribed_text, formatted_data, gr.Button(interactive=True), gr.Button(interactive=False)
            
        except Exception as e:
            self.is_recording = False
            return f"❌ Error stopping recording: {str(e)}", "", "", gr.Button(interactive=True), gr.Button(interactive=False)
    
    def stop_recording_only_speech(self):
        """
        Stop recording and return only transcribed text (for only speech input)
        
        Returns:
            tuple: (status_message, transcribed_text, start_button_state, stop_button_state)
        """
        if not self.is_recording:
            return "⚠️ Not currently recording", "", gr.Button(interactive=True), gr.Button(interactive=False)
        
        try:
            # Stop recording
            self.is_recording = False
            
            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2)
            
            # Get transcribed text using Whisper
            transcribed_text = self.nurse_agent.stop_whisper_recording()
            
            if transcribed_text.startswith("Error") or transcribed_text.startswith("Could not") or transcribed_text.startswith("No"):
                return f"❌ Recording failed: {transcribed_text}", "", gr.Button(interactive=True), gr.Button(interactive=False)
            
            status_msg = f"✅ Recording completed with Whisper!\n📊 Captured {len(transcribed_text)} characters\n🎙️ Ready for direct diagnosis"
            
            return status_msg, transcribed_text, gr.Button(interactive=True), gr.Button(interactive=False)
            
        except Exception as e:
            self.is_recording = False
            return f"❌ Error stopping recording: {str(e)}", "", gr.Button(interactive=True), gr.Button(interactive=False)
    
    def process_speech_input(self, transcribed_text):
        """
        Process transcribed text to generate diagnosis
        
        Args:
            transcribed_text (str): The transcribed speech
            
        Returns:
            str: Diagnosis and next steps
        """
        if not self.initialized:
            return "❌ System not initialized properly"
        
        if not transcribed_text:
            return "❌ No transcribed text to process"
        
        try:
            # Extract patient data from speech
            patient_data = self.nurse_agent.extract_patient_data_from_speech(transcribed_text)
            
            # Format data for LLM
            formatted_data = self.nurse_agent.format_patient_data_for_llm(patient_data)
            formatted_data += f"\n\nORIGINAL PATIENT DESCRIPTION:\n{transcribed_text}"
            
            # Generate diagnosis and next steps
            diagnosis = self.nurse_agent.generate_diagnosis_and_next_steps(formatted_data)
            
            return diagnosis
            
        except Exception as e:
            return f"❌ Error processing speech: {str(e)}"
    
    def process_only_speech_input(self, transcribed_text):
        """
        Process transcribed text directly for diagnosis without data extraction
        
        Args:
            transcribed_text (str): The raw transcribed speech
            
        Returns:
            str: Direct diagnosis from GPT
        """
        if not self.initialized:
            return "❌ System not initialized properly"
        
        if not transcribed_text:
            return "❌ No transcribed text to process"
        
        try:
            # Create a simple prompt for direct diagnosis
            patient_description = f"""
PATIENT DESCRIPTION (Raw Speech):
{transcribed_text}

Date of Assessment: {datetime.now().strftime("%Y-%m-%d")}
"""
            
            # Generate diagnosis directly from raw speech
            diagnosis = self.nurse_agent.generate_diagnosis_and_next_steps(patient_description)
            
            return diagnosis
            
        except Exception as e:
            return f"❌ Error processing speech: {str(e)}"
    
    def process_form_input(self, name, age, gender, fever_present, duration, onset, 
                          fever_frequency, fever_timing, max_temp, chills, sweating,
                          fatigue, headache, muscle_pain, joint_pain, rash, cough,
                          sore_throat, nasal_discharge, abdominal_pain, nausea_vomiting,
                          diarrhea, urinary_symptoms, fever_progression, chronic_illnesses,
                          allergies, current_medications, smoking, alcohol, travel_history,
                          additional_notes):
        """
        Process form input data to generate diagnosis
        
        Returns:
            str: Diagnosis and next steps
        """
        if not self.initialized:
            return "❌ System not initialized properly"
        
        try:
            # Create patient data dictionary
            patient_data = {
                'name': name or 'Not provided',
                'age': age or 'Not provided',
                'gender': gender or 'Not provided',
                'date': datetime.now().strftime("%Y-%m-%d"),
                'fever_present': fever_present,
                'duration': duration or 'Not provided',
                'onset': onset,
                'fever_frequency': fever_frequency,
                'fever_timing': fever_timing,
                'max_temperature': max_temp or 'Not provided',
                'chills_shivering': chills,
                'sweating': sweating,
                'fatigue': fatigue,
                'headache': headache,
                'muscle_pain': muscle_pain,
                'joint_pain': joint_pain,
                'rash': rash,
                'cough': cough,
                'sore_throat': sore_throat,
                'nasal_discharge': nasal_discharge,
                'abdominal_pain': abdominal_pain,
                'nausea_vomiting': nausea_vomiting,
                'diarrhea': diarrhea,
                'urinary_symptoms': urinary_symptoms,
                'fever_progression': fever_progression,
                'chronic_illnesses': chronic_illnesses or 'None',
                'allergies': allergies or 'None',
                'current_medications': current_medications or 'None',
                'smoking': smoking,
                'alcohol': alcohol,
                'travel_history': travel_history or 'None',
                'additional_notes': additional_notes or 'None'
            }
            
            # Format data for LLM
            formatted_data = self.nurse_agent.format_patient_data_for_llm(patient_data)
            
            # Generate diagnosis and next steps
            diagnosis = self.nurse_agent.generate_diagnosis_and_next_steps(formatted_data)
            
            return diagnosis
            
        except Exception as e:
            return f"❌ Error processing form data: {str(e)}"

def create_ui():
    """Create and configure the Gradio interface"""
    
    # Initialize the UI class
    ui = NurseAgentUI()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        color: #2c5aa0;
        margin-bottom: 20px;
    }
    .recording-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .form-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .record-btn {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24) !important;
        color: white !important;
        font-size: 18px !important;
        padding: 15px 30px !important;
        border-radius: 25px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(238, 90, 36, 0.3) !important;
    }
    .record-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(238, 90, 36, 0.4) !important;
    }
    .stop-btn {
        background: linear-gradient(45deg, #fd79a8, #e84393) !important;
        color: white !important;
        font-size: 18px !important;
        padding: 15px 30px !important;
        border-radius: 25px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(232, 67, 147, 0.3) !important;
    }
    .stop-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(232, 67, 147, 0.4) !important;
    }
    """
    
    with gr.Blocks(css=css, title="Fever Proforma Nurse Agent") as demo:
        gr.HTML("""
        <div class="main-header">
            <h1>🏥 Fever Proforma Nurse Agent</h1>
            <h3>AI-Powered Medical Assessment System</h3>
            <p>🎤 Enhanced with OpenAI Whisper for superior speech recognition</p>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            # Speech Input Tab
            with gr.Tab("🎤 Speech Input", elem_classes="recording-section"):
                gr.Markdown("### Record Patient Information with Whisper AI")
                gr.Markdown("""
                **Instructions for Recording:**
                - Click **START RECORDING** to begin capturing audio with OpenAI Whisper
                - Speak naturally about the patient's condition, symptoms, and medical history
                - Include: name, age, symptoms, duration of illness, medical history, medications
                - Click **STOP RECORDING** when you're finished speaking
                - Whisper AI will transcribe everything you say with high accuracy
                - System will extract structured data and generate diagnosis
                """)
                
                with gr.Row():
                    start_btn = gr.Button(
                        "🎤 START RECORDING (Whisper)", 
                        variant="primary",
                        elem_classes="record-btn",
                        scale=1
                    )
                    stop_btn = gr.Button(
                        "⏹️ STOP RECORDING", 
                        variant="secondary",
                        elem_classes="stop-btn",
                        interactive=False,
                        scale=1
                    )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        recording_status = gr.Textbox(
                            label="Recording Status",
                            placeholder="Ready to record...",
                            interactive=False,
                            lines=3
                        )
                    
                with gr.Row():
                    transcribed_text = gr.Textbox(
                        label="Transcribed Speech",
                        placeholder="Transcribed text will appear here...",
                        lines=5,
                        max_lines=10
                    )
                
                with gr.Row():
                    patient_data_display = gr.Textbox(
                        label="Extracted Patient Data",
                        placeholder="Structured patient data will appear here...",
                        lines=10,
                        max_lines=20
                    )
                
                process_speech_btn = gr.Button("🔄 Generate Diagnosis", variant="secondary")
                
                speech_diagnosis = gr.Textbox(
                    label="AI Diagnosis and Next Steps",
                    placeholder="Diagnosis will appear here after processing...",
                    lines=10,
                    max_lines=20
                )
            
            # Only Speech Input Tab
            with gr.Tab("🗣️ Only Speech Input", elem_classes="recording-section"):
                gr.Markdown("### Direct Speech-to-Diagnosis with Whisper")
                gr.Markdown("""
                **Simple Speech Input:**
                - Click **START RECORDING** to capture your speech with Whisper
                - Describe the patient's condition in your own words - no structure needed
                - Click **STOP RECORDING** when finished
                - GPT will analyze your raw speech directly for medical diagnosis
                - No data extraction - just natural conversation to diagnosis
                """)
                
                with gr.Row():
                    only_start_btn = gr.Button(
                        "🎤 START RECORDING (Direct)", 
                        variant="primary",
                        elem_classes="record-btn",
                        scale=1
                    )
                    only_stop_btn = gr.Button(
                        "⏹️ STOP RECORDING", 
                        variant="secondary",
                        elem_classes="stop-btn",
                        interactive=False,
                        scale=1
                    )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        only_recording_status = gr.Textbox(
                            label="Recording Status",
                            placeholder="Ready to record...",
                            interactive=False,
                            lines=3
                        )
                    
                with gr.Row():
                    only_transcribed_text = gr.Textbox(
                        label="Raw Speech Transcription",
                        placeholder="Your spoken words will appear here...",
                        lines=8,
                        max_lines=15
                    )
                
                only_process_btn = gr.Button("🔄 Get Direct Diagnosis", variant="secondary")
                
                only_diagnosis = gr.Textbox(
                    label="Direct AI Diagnosis",
                    placeholder="Direct diagnosis will appear here...",
                    lines=12,
                    max_lines=25
                )
            
            # Form Input Tab
            with gr.Tab("📝 Form Input", elem_classes="form-section"):
                gr.Markdown("### Manual Patient Data Entry")
                
                with gr.Row():
                    with gr.Column():
                        name = gr.Textbox(label="Patient Name", placeholder="Enter patient name")
                        age = gr.Textbox(label="Age", placeholder="Enter age")
                        gender = gr.Dropdown(choices=["Male", "Female", "Other"], label="Gender")
                    
                    with gr.Column():
                        fever_present = gr.Radio(choices=["Yes", "No"], label="Fever Present?", value="Yes")
                        duration = gr.Textbox(label="Duration", placeholder="e.g., 3 days, 1 week")
                        onset = gr.Dropdown(choices=["Sudden", "Gradual"], label="Onset")
                
                with gr.Row():
                    with gr.Column():
                        fever_frequency = gr.Dropdown(
                            choices=["Constantly present", "Comes and goes"], 
                            label="Fever Frequency"
                        )
                        fever_timing = gr.Dropdown(
                            choices=["Day", "Night", "Both"], 
                            label="Fever Timing"
                        )
                        max_temp = gr.Textbox(label="Maximum Temperature", placeholder="e.g., 102°F, 39°C")
                    
                    with gr.Column():
                        fever_progression = gr.Dropdown(
                            choices=["Improving", "Worsening", "Unchanged"], 
                            label="Fever Progression"
                        )
                
                gr.Markdown("#### Associated Symptoms")
                with gr.Row():
                    with gr.Column():
                        chills = gr.Radio(choices=["Yes", "No"], label="Chills/Shivering", value="No")
                        sweating = gr.Radio(choices=["Yes", "No"], label="Sweating", value="No")
                        fatigue = gr.Radio(choices=["Yes", "No"], label="Fatigue", value="No")
                        headache = gr.Radio(choices=["Yes", "No"], label="Headache", value="No")
                        muscle_pain = gr.Radio(choices=["Yes", "No"], label="Muscle Pain", value="No")
                    
                    with gr.Column():
                        joint_pain = gr.Radio(choices=["Yes", "No"], label="Joint Pain", value="No")
                        rash = gr.Radio(choices=["Yes", "No"], label="Rash", value="No")
                        cough = gr.Radio(choices=["Yes", "No"], label="Cough", value="No")
                        sore_throat = gr.Radio(choices=["Yes", "No"], label="Sore Throat", value="No")
                        nasal_discharge = gr.Radio(choices=["Yes", "No"], label="Nasal Discharge", value="No")
                    
                    with gr.Column():
                        abdominal_pain = gr.Radio(choices=["Yes", "No"], label="Abdominal Pain", value="No")
                        nausea_vomiting = gr.Radio(choices=["Yes", "No"], label="Nausea/Vomiting", value="No")
                        diarrhea = gr.Radio(choices=["Yes", "No"], label="Diarrhea", value="No")
                        urinary_symptoms = gr.Radio(choices=["Yes", "No"], label="Urinary Symptoms", value="No")
                
                gr.Markdown("#### Medical History")
                with gr.Row():
                    with gr.Column():
                        chronic_illnesses = gr.Textbox(
                            label="Chronic Illnesses", 
                            placeholder="Diabetes, hypertension, etc."
                        )
                        allergies = gr.Textbox(label="Allergies", placeholder="Known allergies")
                        current_medications = gr.Textbox(
                            label="Current Medications", 
                            placeholder="List current medications"
                        )
                    
                    with gr.Column():
                        smoking = gr.Radio(choices=["Yes", "No"], label="Smoking", value="No")
                        alcohol = gr.Radio(choices=["Yes", "No"], label="Alcohol", value="No")
                        travel_history = gr.Textbox(
                            label="Recent Travel History", 
                            placeholder="Any recent travel"
                        )
                
                additional_notes = gr.Textbox(
                    label="Additional Notes", 
                    placeholder="Any other relevant information",
                    lines=3
                )
                
                submit_form_btn = gr.Button("🔄 Generate Diagnosis", variant="primary")
                
                form_diagnosis = gr.Textbox(
                    label="AI Diagnosis and Next Steps",
                    placeholder="Diagnosis will appear here after processing...",
                    lines=10,
                    max_lines=20
                )
        
        # Event handlers
        start_btn.click(
            fn=ui.start_recording,
            inputs=[],
            outputs=[recording_status, start_btn, stop_btn]
        )
        
        stop_btn.click(
            fn=ui.stop_recording,
            inputs=[],
            outputs=[recording_status, transcribed_text, patient_data_display, start_btn, stop_btn]
        )
        
        process_speech_btn.click(
            fn=ui.process_speech_input,
            inputs=[transcribed_text],
            outputs=[speech_diagnosis]
        )
        
        # Event handlers for Only Speech Input tab
        only_start_btn.click(
            fn=ui.start_recording,
            inputs=[],
            outputs=[only_recording_status, only_start_btn, only_stop_btn]
        )
        
        only_stop_btn.click(
            fn=ui.stop_recording_only_speech,
            inputs=[],
            outputs=[only_recording_status, only_transcribed_text, only_start_btn, only_stop_btn]
        )
        
        only_process_btn.click(
            fn=ui.process_only_speech_input,
            inputs=[only_transcribed_text],
            outputs=[only_diagnosis]
        )
        
        submit_form_btn.click(
            fn=ui.process_form_input,
            inputs=[
                name, age, gender, fever_present, duration, onset,
                fever_frequency, fever_timing, max_temp, chills, sweating,
                fatigue, headache, muscle_pain, joint_pain, rash, cough,
                sore_throat, nasal_discharge, abdominal_pain, nausea_vomiting,
                diarrhea, urinary_symptoms, fever_progression, chronic_illnesses,
                allergies, current_medications, smoking, alcohol, travel_history,
                additional_notes
            ],
            outputs=[form_diagnosis]
        )
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Fever Proforma Nurse Agent Web UI...")
    
    # Create and launch the interface
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,  # Use a different port
        share=True,
        show_error=True,
        quiet=False
    )
