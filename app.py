import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
from audio_recorder_streamlit import audio_recorder
import streamlit.components.v1 as components

# Created by Danielle Bagaforo Meer (Algorex)
# LinkedIn : https://www.linkedin.com/in/algorexph/

warnings.filterwarnings("ignore")
st.set_page_config(page_title="CertAI by Generative Labs", page_icon=":book:", layout="wide")

with st.sidebar :
    st.title("Generative Labs")
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==51):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "Question_1", "Question_2", "Question_3", "Question_4", "Question_5", "Question_6", "Question_7", "Question_8", "Question_9", "Question_10"],
        icons = ['house', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if options == "Home" :
  st.title("Introducing CertAI: Revolutionizing Multiple-Choice Certification Exams")
  st.write("We are thrilled to introduce CertAI, your cutting-edge AI companion designed to transform the way you approach multiple-choice certification exams. CertAI brings a new level of insight to your exam preparation by not only analyzing your answers but also incorporating a unique explanation component.")

  st.write("# What is CertAI?")
  st.write("CertAI is an advanced AI-powered tool that specializes in enhancing your certification exam experience. By focusing on the reasoning behind your answers, CertAI ensures that you don’t just know the right answers, but also understand why they are correct.")

  st.write("# How Does CertAI Help You?")
  st.write("1. **Answer Analysis**: CertAI evaluates your selected answers, offering feedback that helps you understand your strengths and areas for improvement.")
  st.write("2. **Explanation Component**: CertAI requires you to provide an explanation for each answer, encouraging deeper understanding and retention of the material.")
  st.write("3. **Tailored Feedback**: Based on your explanations, CertAI offers insights into your reasoning process, helping you refine your approach and achieve better results.")

  st.write("# Why Choose CertAI?")
  st.write("- **Deep Understanding**: CertAI emphasizes the importance of understanding concepts, not just memorizing answers.")
  st.write("- **Interactive Learning**: Engage more actively with the material by explaining your answers, leading to better comprehension and exam performance.")
  st.write("- **Personalized Guidance**: CertAI adapts to your learning style, providing personalized feedback to help you improve where it matters most.")
  st.write("- **AI-Powered Efficiency**: Maximize your study sessions with AI-driven analysis and feedback that focuses on both accuracy and understanding.")

  st.write("# Elevate Your Certification Exam Preparation")
  st.write("CertAI is here to take your exam preparation to the next level. By integrating answer explanations into the process, CertAI ensures that you gain a deeper understanding of the material, leading to greater success in your certification exams. Experience the future of exam preparation with CertAI!")

# Question 1
elif options == "Question_1" :

     html_content = """
<div style="border: 2px solid #333; border-radius: 8px; padding: 20px; width: 400px; margin: 20px auto; background-color: #f9f9f9; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
    <div style="font-size: 24px, font-weight: bold; margin-bottom: 10px; color: #000;"> Question 1</div>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #000;">What technique can be used in prompt engineering to ensure the generation of factually accurate responses?</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">A. Bias correction</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">B. Fine-tuning</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">C. Adversarial training</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">D. Data augmentation</div>
</div>
"""


     components.html(html_content, height=300)

     col1, col2, col3 = st.columns(3)

     with col1:
          st.write(' ')

     with col2:
          # Record audio
          audio_data = audio_recorder()

          # If audio is recorded, display the audio player
          if audio_data:
             st.audio(audio_data, format="audio/wav")

          # Save the audio file
          with open("recorded_audio.wav", "wb") as f:
               f.write(audio_data)

     with col3:
          st.write(' ')

     audio_file_path = "recorded_audio.wav"

     if os.path.exists(audio_file_path):
        if st.button("Submit Recording"):
           audio_file= open("recorded_audio.wav", "rb")
           transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
           st.write("### Transcription :")
           st.write(transcription.text)
           dataframed = pd.read_csv('Dataset/ChatGPT_Prompt_Engineering_Cert_Basic_Modified.csv', encoding='ISO-8859-1')
           dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
           documents = dataframed['combined'].tolist()
           System_Prompt = """
You are an ai assistant that will take in a multiple choice questions alongside the user's final answer and explanation.

The User's answer needs to provide the final answer alongside their explanation on why that is the answer. The grading of each question must be 0.7 for the final answer and 0.3 for the explanation. If the User is wrong, the ai will provide the correct answer and explanation on why it is the correct answer. Else if the user is correct, the AI will acknowledge it accordingly. The AI will evaluate the user's answer and provide the user with feedback.


To Score the Explanation of the User per question, you will be providing a general criteria in which will grant the user 0.1 of the 0.3 points for explanation if they are able to achieve atleast one of it.

General Criteria :
Accuracy of Information: The explanation should accurately reflect established knowledge about AI and machine learning, specifically regarding prompt engineering.

Clarity of Explanation: The explanation should be clear and understandable, avoiding technical jargon where possible or defining such terms when used.

Relevance to the Question: Ensure that the explanation directly addresses the question asked, linking back to the specific techniques or concepts referenced in the question.

Depth of Insight: Look for explanations that provide depth beyond a superficial understanding, indicating critical thinking or application of the concept.

Evidence of Practical Understanding: Favor explanations that include practical examples or real-world applications of the concept, demonstrating how it can be applied outside theoretical contexts.

Then for the 0.2 points of the 0.3 Points for explanation will be coming from the provided Explanation Criteria in the context for each question. A user who atleast fulfills one explanation criteria will be granted 0.1 points, while a user who fulfills atleast two explanation criteria will be granted 0.2 points.

Question :
What technique can be used in prompt engineering to ensure the generation of factually accurate responses?
A. Bias correction
B. Fine-tuning
C. Adversarial training
D. Data augmentation

""" + "Explanation Criteria : \n" + dataframed['Explanation_Criteria'][0] + "\n" + "Correct Answer : " + dataframed['Correct Answer'][0]
           
           struct = [ {"role": "system", "content": System_Prompt} ]

           user_answer = "User Answer and Explanation : " + transcription.text
           struct.append({"role" : "user", "content" : user_answer})
           chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct, temperature=0.2, max_tokens=3500, top_p=1, frequency_penalty=0, presence_penalty=0)
           response = chat.choices[0].message.content
           st.write("Feedback : ")
           st.write(response)
     else:
           st.write("No recorded audio available. Please record audio first.")

# Question 2
elif options == "Question_2" :

     html_content = """
<div style="border: 2px solid #333; border-radius: 8px; padding: 20px; width: 400px; margin: 20px auto; background-color: #f9f9f9; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
    <div style="font-size: 24px, font-weight: bold; margin-bottom: 10px; color: #000;"> Question 2</div>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #000;"> How does adjusting the temperature parameter affect the responses generated by ChatGPT?</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">A. Increases randomness</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">B. Reduces response length</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">C. Enhances response coherence</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">D. Introduces grammatical errors</div>
</div>
"""


     components.html(html_content, height=300)

     col1, col2, col3 = st.columns(3)

     with col1:
          st.write(' ')

     with col2:
          # Record audio
          audio_data = audio_recorder()

          # If audio is recorded, display the audio player
          if audio_data:
             st.audio(audio_data, format="audio/wav")

          # Save the audio file
          with open("recorded_audio.wav", "wb") as f:
               f.write(audio_data)

     with col3:
          st.write(' ')

     audio_file_path = "recorded_audio.wav"

     if os.path.exists(audio_file_path):
        if st.button("Submit Recording"):
           audio_file= open("recorded_audio.wav", "rb")
           transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
           st.write("### Transcription :")
           st.write(transcription.text)
           dataframed = pd.read_csv('Dataset/ChatGPT_Prompt_Engineering_Cert_Basic_Modified.csv', encoding='ISO-8859-1')
           dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
           documents = dataframed['combined'].tolist()
           System_Prompt = """
You are an ai assistant that will take in a multiple choice questions alongside the user's final answer and explanation.

The User's answer needs to provide the final answer alongside their explanation on why that is the answer. The grading of each question must be 0.7 for the final answer and 0.3 for the explanation. If the User is wrong, the ai will provide the correct answer and explanation on why it is the correct answer. Else if the user is correct, the AI will acknowledge it accordingly. The AI will evaluate the user's answer and provide the user with feedback.


To Score the Explanation of the User per question, you will be providing a general criteria in which will grant the user 0.1 of the 0.3 points for explanation if they are able to achieve atleast one of it.

General Criteria :
Accuracy of Information: The explanation should accurately reflect established knowledge about AI and machine learning, specifically regarding prompt engineering.

Clarity of Explanation: The explanation should be clear and understandable, avoiding technical jargon where possible or defining such terms when used.

Relevance to the Question: Ensure that the explanation directly addresses the question asked, linking back to the specific techniques or concepts referenced in the question.

Depth of Insight: Look for explanations that provide depth beyond a superficial understanding, indicating critical thinking or application of the concept.

Evidence of Practical Understanding: Favor explanations that include practical examples or real-world applications of the concept, demonstrating how it can be applied outside theoretical contexts.

Then for the 0.2 points of the 0.3 Points for explanation will be coming from the provided Explanation Criteria in the context for each question. A user who atleast fulfills one explanation criteria will be granted 0.1 points, while a user who fulfills atleast two explanation criteria will be granted 0.2 points.

Question :
How does adjusting the temperature parameter affect the responses generated by ChatGPT?
A. Increases randomness
B. Reduces response length
C. Enhances response coherence
D. Introduces grammatical errors

""" + "Explanation Criteria : \n" + dataframed['Explanation_Criteria'][1] + "\n" + "Correct Answer : " + dataframed['Correct Answer'][1]
           
           struct = [ {"role": "system", "content": System_Prompt} ]

           user_answer = "User Answer and Explanation : " + transcription.text
           struct.append({"role" : "user", "content" : user_answer})
           chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct, temperature=0.2, max_tokens=3500, top_p=1, frequency_penalty=0, presence_penalty=0)
           response = chat.choices[0].message.content
           st.write("Feedback : ")
           st.write(response)
     else:
           st.write("No recorded audio available. Please record audio first.")

# Question 3
elif options == "Question_3" :

     html_content = """
<div style="border: 2px solid #333; border-radius: 8px; padding: 20px; width: 400px; margin: 20px auto; background-color: #f9f9f9; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
    <div style="font-size: 24px, font-weight: bold; margin-bottom: 10px; color: #000;"> Question 3</div>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #000;"> Which technique is used in prompt engineering to enhance the model's ability to generate responses in multiple languages?</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">A. Multilingual fine-tuning</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">B. Language-specific data augmentation</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">C. Translation mapping</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">D. Cross-lingual adaptation</div>
</div>
"""


     components.html(html_content, height=300)

     col1, col2, col3 = st.columns(3)

     with col1:
          st.write(' ')

     with col2:
          # Record audio
          audio_data = audio_recorder()

          # If audio is recorded, display the audio player
          if audio_data:
             st.audio(audio_data, format="audio/wav")

          # Save the audio file
          with open("recorded_audio.wav", "wb") as f:
               f.write(audio_data)

     with col3:
          st.write(' ')

     audio_file_path = "recorded_audio.wav"

     if os.path.exists(audio_file_path):
        if st.button("Submit Recording"):
           audio_file= open("recorded_audio.wav", "rb")
           transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
           st.write("### Transcription :")
           st.write(transcription.text)
           dataframed = pd.read_csv('Dataset/ChatGPT_Prompt_Engineering_Cert_Basic_Modified.csv', encoding='ISO-8859-1')
           dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
           documents = dataframed['combined'].tolist()
           System_Prompt = """
You are an ai assistant that will take in a multiple choice questions alongside the user's final answer and explanation.

The User's answer needs to provide the final answer alongside their explanation on why that is the answer. The grading of each question must be 0.7 for the final answer and 0.3 for the explanation. If the User is wrong, the ai will provide the correct answer and explanation on why it is the correct answer. Else if the user is correct, the AI will acknowledge it accordingly. The AI will evaluate the user's answer and provide the user with feedback.


To Score the Explanation of the User per question, you will be providing a general criteria in which will grant the user 0.1 of the 0.3 points for explanation if they are able to achieve atleast one of it.

General Criteria :
Accuracy of Information: The explanation should accurately reflect established knowledge about AI and machine learning, specifically regarding prompt engineering.

Clarity of Explanation: The explanation should be clear and understandable, avoiding technical jargon where possible or defining such terms when used.

Relevance to the Question: Ensure that the explanation directly addresses the question asked, linking back to the specific techniques or concepts referenced in the question.

Depth of Insight: Look for explanations that provide depth beyond a superficial understanding, indicating critical thinking or application of the concept.

Evidence of Practical Understanding: Favor explanations that include practical examples or real-world applications of the concept, demonstrating how it can be applied outside theoretical contexts.

Then for the 0.2 points of the 0.3 Points for explanation will be coming from the provided Explanation Criteria in the context for each question. A user who atleast fulfills one explanation criteria will be granted 0.1 points, while a user who fulfills atleast two explanation criteria will be granted 0.2 points.

Question :
Which technique is used in prompt engineering to enhance the model's ability to generate responses in multiple languages?
A. Multilingual fine-tuning
B. Language-specific data augmentation
C. Translation mapping
D. Cross-lingual adaptation

""" + "Explanation Criteria : \n" + dataframed['Explanation_Criteria'][2] + "\n" + "Correct Answer : " + dataframed['Correct Answer'][2]
           
           struct = [ {"role": "system", "content": System_Prompt} ]

           user_answer = "User Answer and Explanation : " + transcription.text
           struct.append({"role" : "user", "content" : user_answer})
           chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct, temperature=0.2, max_tokens=3500, top_p=1, frequency_penalty=0, presence_penalty=0)
           response = chat.choices[0].message.content
           st.write("Feedback : ")
           st.write(response)
     else:
           st.write("No recorded audio available. Please record audio first.")

# Question 4
elif options == "Question_4" :

     html_content = """
<div style="border: 2px solid #333; border-radius: 8px; padding: 20px; width: 400px; margin: 20px auto; background-color: #f9f9f9; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
    <div style="font-size: 24px, font-weight: bold; margin-bottom: 10px; color: #000;"> Question 4</div>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #000;"> In prompt engineering, what is the purpose of using prefix prompts?</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">A. Provide context for response generation</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">B. Ensure grammatical correctness</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">C. Control response length</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">D. Enhance model training</div>
</div>
"""


     components.html(html_content, height=300)

     col1, col2, col3 = st.columns(3)

     with col1:
          st.write(' ')

     with col2:
          # Record audio
          audio_data = audio_recorder()

          # If audio is recorded, display the audio player
          if audio_data:
             st.audio(audio_data, format="audio/wav")

          # Save the audio file
          with open("recorded_audio.wav", "wb") as f:
               f.write(audio_data)

     with col3:
          st.write(' ')

     audio_file_path = "recorded_audio.wav"

     if os.path.exists(audio_file_path):
        if st.button("Submit Recording"):
           audio_file= open("recorded_audio.wav", "rb")
           transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
           st.write("### Transcription :")
           st.write(transcription.text)
           dataframed = pd.read_csv('Dataset/ChatGPT_Prompt_Engineering_Cert_Basic_Modified.csv', encoding='ISO-8859-1')
           dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
           documents = dataframed['combined'].tolist()
           System_Prompt = """
You are an ai assistant that will take in a multiple choice questions alongside the user's final answer and explanation.

The User's answer needs to provide the final answer alongside their explanation on why that is the answer. The grading of each question must be 0.7 for the final answer and 0.3 for the explanation. If the User is wrong, the ai will provide the correct answer and explanation on why it is the correct answer. Else if the user is correct, the AI will acknowledge it accordingly. The AI will evaluate the user's answer and provide the user with feedback.


To Score the Explanation of the User per question, you will be providing a general criteria in which will grant the user 0.1 of the 0.3 points for explanation if they are able to achieve atleast one of it.

General Criteria :
Accuracy of Information: The explanation should accurately reflect established knowledge about AI and machine learning, specifically regarding prompt engineering.

Clarity of Explanation: The explanation should be clear and understandable, avoiding technical jargon where possible or defining such terms when used.

Relevance to the Question: Ensure that the explanation directly addresses the question asked, linking back to the specific techniques or concepts referenced in the question.

Depth of Insight: Look for explanations that provide depth beyond a superficial understanding, indicating critical thinking or application of the concept.

Evidence of Practical Understanding: Favor explanations that include practical examples or real-world applications of the concept, demonstrating how it can be applied outside theoretical contexts.

Then for the 0.2 points of the 0.3 Points for explanation will be coming from the provided Explanation Criteria in the context for each question. A user who atleast fulfills one explanation criteria will be granted 0.1 points, while a user who fulfills atleast two explanation criteria will be granted 0.2 points.

Question :
In prompt engineering, what is the purpose of using prefix prompts?
A. Provide context for response generation
B. Ensure grammatical correctness
C. Control response length
D. Enhance model training

""" + "Explanation Criteria : \n" + dataframed['Explanation_Criteria'][3] + "\n" + "Correct Answer : " + dataframed['Correct Answer'][3]
           
           struct = [ {"role": "system", "content": System_Prompt} ]

           user_answer = "User Answer and Explanation : " + transcription.text
           struct.append({"role" : "user", "content" : user_answer})
           chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct, temperature=0.2, max_tokens=3500, top_p=1, frequency_penalty=0, presence_penalty=0)
           response = chat.choices[0].message.content
           st.write("Feedback : ")
           st.write(response)
     else:
           st.write("No recorded audio available. Please record audio first.")

# Question 5
elif options == "Question_5" :

     html_content = """
<div style="border: 2px solid #333; border-radius: 8px; padding: 20px; width: 400px; margin: 20px auto; background-color: #f9f9f9; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
    <div style="font-size: 24px, font-weight: bold; margin-bottom: 10px; color: #000;"> Question 5</div>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #000;"> Which parameter in OpenAI's GPT models controls the level of "novelty" in generated responses?</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">A. Top-p sampling</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">B. Nucleus sampling</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">C. Diversity penalty</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">D. Response uniqueness</div>
</div>
"""


     components.html(html_content, height=300)

     col1, col2, col3 = st.columns(3)

     with col1:
          st.write(' ')

     with col2:
          # Record audio
          audio_data = audio_recorder()

          # If audio is recorded, display the audio player
          if audio_data:
             st.audio(audio_data, format="audio/wav")

          # Save the audio file
          with open("recorded_audio.wav", "wb") as f:
               f.write(audio_data)

     with col3:
          st.write(' ')

     audio_file_path = "recorded_audio.wav"

     if os.path.exists(audio_file_path):
        if st.button("Submit Recording"):
           audio_file= open("recorded_audio.wav", "rb")
           transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
           st.write("### Transcription :")
           st.write(transcription.text)
           dataframed = pd.read_csv('Dataset/ChatGPT_Prompt_Engineering_Cert_Basic_Modified.csv', encoding='ISO-8859-1')
           dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
           documents = dataframed['combined'].tolist()
           System_Prompt = """
You are an ai assistant that will take in a multiple choice questions alongside the user's final answer and explanation.

The User's answer needs to provide the final answer alongside their explanation on why that is the answer. The grading of each question must be 0.7 for the final answer and 0.3 for the explanation. If the User is wrong, the ai will provide the correct answer and explanation on why it is the correct answer. Else if the user is correct, the AI will acknowledge it accordingly. The AI will evaluate the user's answer and provide the user with feedback.


To Score the Explanation of the User per question, you will be providing a general criteria in which will grant the user 0.1 of the 0.3 points for explanation if they are able to achieve atleast one of it.

General Criteria :
Accuracy of Information: The explanation should accurately reflect established knowledge about AI and machine learning, specifically regarding prompt engineering.

Clarity of Explanation: The explanation should be clear and understandable, avoiding technical jargon where possible or defining such terms when used.

Relevance to the Question: Ensure that the explanation directly addresses the question asked, linking back to the specific techniques or concepts referenced in the question.

Depth of Insight: Look for explanations that provide depth beyond a superficial understanding, indicating critical thinking or application of the concept.

Evidence of Practical Understanding: Favor explanations that include practical examples or real-world applications of the concept, demonstrating how it can be applied outside theoretical contexts.

Then for the 0.2 points of the 0.3 Points for explanation will be coming from the provided Explanation Criteria in the context for each question. A user who atleast fulfills one explanation criteria will be granted 0.1 points, while a user who fulfills atleast two explanation criteria will be granted 0.2 points.

Question :
Which parameter in OpenAI's GPT models controls the level of "novelty" in generated responses?
A. Top-p sampling
B. Nucleus sampling
C. Diversity penalty
D. Response uniqueness

""" + "Explanation Criteria : \n" + dataframed['Explanation_Criteria'][4] + "\n" + "Correct Answer : " + dataframed['Correct Answer'][4]
           
           struct = [ {"role": "system", "content": System_Prompt} ]

           user_answer = "User Answer and Explanation : " + transcription.text
           struct.append({"role" : "user", "content" : user_answer})
           chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct, temperature=0.2, max_tokens=3500, top_p=1, frequency_penalty=0, presence_penalty=0)
           response = chat.choices[0].message.content
           st.write("Feedback : ")
           st.write(response)
     else:
           st.write("No recorded audio available. Please record audio first.")

# Question 6
elif options == "Question_6" :

     html_content = """
<div style="border: 2px solid #333; border-radius: 8px; padding: 20px; width: 400px; margin: 20px auto; background-color: #f9f9f9; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
    <div style="font-size: 24px, font-weight: bold; margin-bottom: 10px; color: #000;"> Question 6</div>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #000;"> What is a potential drawback of using large-scale data augmentation in training ChatGPT models?</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">A. Increased computational costs</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">B. Reduced model generalization</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">C. Lower response accuracy</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">D. Limited training data availability</div>
</div>
"""


     components.html(html_content, height=300)

     col1, col2, col3 = st.columns(3)

     with col1:
          st.write(' ')

     with col2:
          # Record audio
          audio_data = audio_recorder()

          # If audio is recorded, display the audio player
          if audio_data:
             st.audio(audio_data, format="audio/wav")

          # Save the audio file
          with open("recorded_audio.wav", "wb") as f:
               f.write(audio_data)

     with col3:
          st.write(' ')

     audio_file_path = "recorded_audio.wav"

     if os.path.exists(audio_file_path):
        if st.button("Submit Recording"):
           audio_file= open("recorded_audio.wav", "rb")
           transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
           st.write("### Transcription :")
           st.write(transcription.text)
           dataframed = pd.read_csv('Dataset/ChatGPT_Prompt_Engineering_Cert_Basic_Modified.csv', encoding='ISO-8859-1')
           dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
           documents = dataframed['combined'].tolist()
           System_Prompt = """
You are an ai assistant that will take in a multiple choice questions alongside the user's final answer and explanation.

The User's answer needs to provide the final answer alongside their explanation on why that is the answer. The grading of each question must be 0.7 for the final answer and 0.3 for the explanation. If the User is wrong, the ai will provide the correct answer and explanation on why it is the correct answer. Else if the user is correct, the AI will acknowledge it accordingly. The AI will evaluate the user's answer and provide the user with feedback.


To Score the Explanation of the User per question, you will be providing a general criteria in which will grant the user 0.1 of the 0.3 points for explanation if they are able to achieve atleast one of it.

General Criteria :
Accuracy of Information: The explanation should accurately reflect established knowledge about AI and machine learning, specifically regarding prompt engineering.

Clarity of Explanation: The explanation should be clear and understandable, avoiding technical jargon where possible or defining such terms when used.

Relevance to the Question: Ensure that the explanation directly addresses the question asked, linking back to the specific techniques or concepts referenced in the question.

Depth of Insight: Look for explanations that provide depth beyond a superficial understanding, indicating critical thinking or application of the concept.

Evidence of Practical Understanding: Favor explanations that include practical examples or real-world applications of the concept, demonstrating how it can be applied outside theoretical contexts.

Then for the 0.2 points of the 0.3 Points for explanation will be coming from the provided Explanation Criteria in the context for each question. A user who atleast fulfills one explanation criteria will be granted 0.1 points, while a user who fulfills atleast two explanation criteria will be granted 0.2 points.

Question :
What is a potential drawback of using large-scale data augmentation in training ChatGPT models?
A. Increased computational costs
B. Reduced model generalization
C. Lower response accuracy
D. Limited training data availability

""" + "Explanation Criteria : \n" + dataframed['Explanation_Criteria'][5] + "\n" + "Correct Answer : " + dataframed['Correct Answer'][5]
           
           struct = [ {"role": "system", "content": System_Prompt} ]

           user_answer = "User Answer and Explanation : " + transcription.text
           struct.append({"role" : "user", "content" : user_answer})
           chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct, temperature=0.2, max_tokens=3500, top_p=1, frequency_penalty=0, presence_penalty=0)
           response = chat.choices[0].message.content
           st.write("Feedback : ")
           st.write(response)
     else:
           st.write("No recorded audio available. Please record audio first.")


# Question 7
elif options == "Question_7" :

     html_content = """
<div style="border: 2px solid #333; border-radius: 8px; padding: 20px; width: 400px; margin: 20px auto; background-color: #f9f9f9; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
    <div style="font-size: 24px, font-weight: bold; margin-bottom: 10px; color: #000;"> Question 7</div>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #000;"> How does adjusting the "presence penalty" parameter affect the relevance of generated responses?</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">A. Increases context awareness</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">B. Reduces repetition in responses</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">C. Enhances response specificity</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">D. Improves response coherence</div>
</div>
"""


     components.html(html_content, height=300)

     col1, col2, col3 = st.columns(3)

     with col1:
          st.write(' ')

     with col2:
          # Record audio
          audio_data = audio_recorder()

          # If audio is recorded, display the audio player
          if audio_data:
             st.audio(audio_data, format="audio/wav")

          # Save the audio file
          with open("recorded_audio.wav", "wb") as f:
               f.write(audio_data)

     with col3:
          st.write(' ')

     audio_file_path = "recorded_audio.wav"

     if os.path.exists(audio_file_path):
        if st.button("Submit Recording"):
           audio_file= open("recorded_audio.wav", "rb")
           transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
           st.write("### Transcription :")
           st.write(transcription.text)
           dataframed = pd.read_csv('Dataset/ChatGPT_Prompt_Engineering_Cert_Basic_Modified.csv', encoding='ISO-8859-1')
           dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
           documents = dataframed['combined'].tolist()
           System_Prompt = """
You are an ai assistant that will take in a multiple choice questions alongside the user's final answer and explanation.

The User's answer needs to provide the final answer alongside their explanation on why that is the answer. The grading of each question must be 0.7 for the final answer and 0.3 for the explanation. If the User is wrong, the ai will provide the correct answer and explanation on why it is the correct answer. Else if the user is correct, the AI will acknowledge it accordingly. The AI will evaluate the user's answer and provide the user with feedback.


To Score the Explanation of the User per question, you will be providing a general criteria in which will grant the user 0.1 of the 0.3 points for explanation if they are able to achieve atleast one of it.

General Criteria :
Accuracy of Information: The explanation should accurately reflect established knowledge about AI and machine learning, specifically regarding prompt engineering.

Clarity of Explanation: The explanation should be clear and understandable, avoiding technical jargon where possible or defining such terms when used.

Relevance to the Question: Ensure that the explanation directly addresses the question asked, linking back to the specific techniques or concepts referenced in the question.

Depth of Insight: Look for explanations that provide depth beyond a superficial understanding, indicating critical thinking or application of the concept.

Evidence of Practical Understanding: Favor explanations that include practical examples or real-world applications of the concept, demonstrating how it can be applied outside theoretical contexts.

Then for the 0.2 points of the 0.3 Points for explanation will be coming from the provided Explanation Criteria in the context for each question. A user who atleast fulfills one explanation criteria will be granted 0.1 points, while a user who fulfills atleast two explanation criteria will be granted 0.2 points.

Question :
How does adjusting the "presence penalty" parameter affect the relevance of generated responses?
A. Increases context awareness
B. Reduces repetition in responses
C. Enhances response specificity
D. Improves response coherence

""" + "Explanation Criteria : \n" + dataframed['Explanation_Criteria'][6] + "\n" + "Correct Answer : " + dataframed['Correct Answer'][6]
           
           struct = [ {"role": "system", "content": System_Prompt} ]

           user_answer = "User Answer and Explanation : " + transcription.text
           struct.append({"role" : "user", "content" : user_answer})
           chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct, temperature=0.2, max_tokens=3500, top_p=1, frequency_penalty=0, presence_penalty=0)
           response = chat.choices[0].message.content
           st.write("Feedback : ")
           st.write(response)
     else:
           st.write("No recorded audio available. Please record audio first.")

# Question 8
elif options == "Question_8" :

     html_content = """
<div style="border: 2px solid #333; border-radius: 8px; padding: 20px; width: 400px; margin: 20px auto; background-color: #f9f9f9; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
    <div style="font-size: 24px, font-weight: bold; margin-bottom: 10px; color: #000;"> Question 8</div>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #000;"> Which technique is used to ensure the ethical use of AI models like ChatGPT in sensitive applications?</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">A. Bias correction</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">B. Fairness regularization</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">C. Accountability framing</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">D. Adversarial training</div>
</div>
"""


     components.html(html_content, height=300)

     col1, col2, col3 = st.columns(3)

     with col1:
          st.write(' ')

     with col2:
          # Record audio
          audio_data = audio_recorder()

          # If audio is recorded, display the audio player
          if audio_data:
             st.audio(audio_data, format="audio/wav")

          # Save the audio file
          with open("recorded_audio.wav", "wb") as f:
               f.write(audio_data)

     with col3:
          st.write(' ')

     audio_file_path = "recorded_audio.wav"

     if os.path.exists(audio_file_path):
        if st.button("Submit Recording"):
           audio_file= open("recorded_audio.wav", "rb")
           transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
           st.write("### Transcription :")
           st.write(transcription.text)
           dataframed = pd.read_csv('Dataset/ChatGPT_Prompt_Engineering_Cert_Basic_Modified.csv', encoding='ISO-8859-1')
           dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
           documents = dataframed['combined'].tolist()
           System_Prompt = """
You are an ai assistant that will take in a multiple choice questions alongside the user's final answer and explanation.

The User's answer needs to provide the final answer alongside their explanation on why that is the answer. The grading of each question must be 0.7 for the final answer and 0.3 for the explanation. If the User is wrong, the ai will provide the correct answer and explanation on why it is the correct answer. Else if the user is correct, the AI will acknowledge it accordingly. The AI will evaluate the user's answer and provide the user with feedback.


To Score the Explanation of the User per question, you will be providing a general criteria in which will grant the user 0.1 of the 0.3 points for explanation if they are able to achieve atleast one of it.

General Criteria :
Accuracy of Information: The explanation should accurately reflect established knowledge about AI and machine learning, specifically regarding prompt engineering.

Clarity of Explanation: The explanation should be clear and understandable, avoiding technical jargon where possible or defining such terms when used.

Relevance to the Question: Ensure that the explanation directly addresses the question asked, linking back to the specific techniques or concepts referenced in the question.

Depth of Insight: Look for explanations that provide depth beyond a superficial understanding, indicating critical thinking or application of the concept.

Evidence of Practical Understanding: Favor explanations that include practical examples or real-world applications of the concept, demonstrating how it can be applied outside theoretical contexts.

Then for the 0.2 points of the 0.3 Points for explanation will be coming from the provided Explanation Criteria in the context for each question. A user who atleast fulfills one explanation criteria will be granted 0.1 points, while a user who fulfills atleast two explanation criteria will be granted 0.2 points.

Question :
Which technique is used to ensure the ethical use of AI models like ChatGPT in sensitive applications?
A. Bias correction
B. Fairness regularization
C. Accountability framing
D. Adversarial training

""" + "Explanation Criteria : \n" + dataframed['Explanation_Criteria'][7] + "\n" + "Correct Answer : " + dataframed['Correct Answer'][7]
           
           struct = [ {"role": "system", "content": System_Prompt} ]

           user_answer = "User Answer and Explanation : " + transcription.text
           struct.append({"role" : "user", "content" : user_answer})
           chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct, temperature=0.2, max_tokens=3500, top_p=1, frequency_penalty=0, presence_penalty=0)
           response = chat.choices[0].message.content
           st.write("Feedback : ")
           st.write(response)
     else:
           st.write("No recorded audio available. Please record audio first.")

# Question 9
elif options == "Question_9" :

     html_content = """
<div style="border: 2px solid #333; border-radius: 8px; padding: 20px; width: 400px; margin: 20px auto; background-color: #f9f9f9; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
    <div style="font-size: 24px, font-weight: bold; margin-bottom: 10px; color: #000;"> Question 9</div>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #000;"> How does fine-tuning affect the responsiveness of ChatGPT in specialized tasks?</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">A. Increases model speed</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">B. Enhances task-specific performance</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">C. Reduces memory usage</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">D.  Improves model scalability</div>
</div>
"""


     components.html(html_content, height=300)

     col1, col2, col3 = st.columns(3)

     with col1:
          st.write(' ')

     with col2:
          # Record audio
          audio_data = audio_recorder()

          # If audio is recorded, display the audio player
          if audio_data:
             st.audio(audio_data, format="audio/wav")

          # Save the audio file
          with open("recorded_audio.wav", "wb") as f:
               f.write(audio_data)

     with col3:
          st.write(' ')

     audio_file_path = "recorded_audio.wav"

     if os.path.exists(audio_file_path):
        if st.button("Submit Recording"):
           audio_file= open("recorded_audio.wav", "rb")
           transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
           st.write("### Transcription :")
           st.write(transcription.text)
           dataframed = pd.read_csv('Dataset/ChatGPT_Prompt_Engineering_Cert_Basic_Modified.csv', encoding='ISO-8859-1')
           dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
           documents = dataframed['combined'].tolist()
           System_Prompt = """
You are an ai assistant that will take in a multiple choice questions alongside the user's final answer and explanation.

The User's answer needs to provide the final answer alongside their explanation on why that is the answer. The grading of each question must be 0.7 for the final answer and 0.3 for the explanation. If the User is wrong, the ai will provide the correct answer and explanation on why it is the correct answer. Else if the user is correct, the AI will acknowledge it accordingly. The AI will evaluate the user's answer and provide the user with feedback.


To Score the Explanation of the User per question, you will be providing a general criteria in which will grant the user 0.1 of the 0.3 points for explanation if they are able to achieve atleast one of it.

General Criteria :
Accuracy of Information: The explanation should accurately reflect established knowledge about AI and machine learning, specifically regarding prompt engineering.

Clarity of Explanation: The explanation should be clear and understandable, avoiding technical jargon where possible or defining such terms when used.

Relevance to the Question: Ensure that the explanation directly addresses the question asked, linking back to the specific techniques or concepts referenced in the question.

Depth of Insight: Look for explanations that provide depth beyond a superficial understanding, indicating critical thinking or application of the concept.

Evidence of Practical Understanding: Favor explanations that include practical examples or real-world applications of the concept, demonstrating how it can be applied outside theoretical contexts.

Then for the 0.2 points of the 0.3 Points for explanation will be coming from the provided Explanation Criteria in the context for each question. A user who atleast fulfills one explanation criteria will be granted 0.1 points, while a user who fulfills atleast two explanation criteria will be granted 0.2 points.

Question :
How does fine-tuning affect the responsiveness of ChatGPT in specialized tasks?
A. Increases model speed
B. Enhances task-specific performance
C. Reduces memory usage
D. Improves model scalability

""" + "Explanation Criteria : \n" + dataframed['Explanation_Criteria'][8] + "\n" + "Correct Answer : " + dataframed['Correct Answer'][8]
           
           struct = [ {"role": "system", "content": System_Prompt} ]

           user_answer = "User Answer and Explanation : " + transcription.text
           struct.append({"role" : "user", "content" : user_answer})
           chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct, temperature=0.2, max_tokens=3500, top_p=1, frequency_penalty=0, presence_penalty=0)
           response = chat.choices[0].message.content
           st.write("Feedback : ")
           st.write(response)
     else:
           st.write("No recorded audio available. Please record audio first.")

# Question 10
elif options == "Question_10" :

     html_content = """
<div style="border: 2px solid #333; border-radius: 8px; padding: 20px; width: 400px; margin: 20px auto; background-color: #f9f9f9; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
    <div style="font-size: 24px, font-weight: bold; margin-bottom: 10px; color: #000;"> Question 10</div>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #000;"> What is a potential benefit of using ensemble techniques with ChatGPT models?</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">A. Increases model interpretability</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">B. Boosts response diversity</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">C. Reduces training time</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">D. Enhances language fluency</div>
</div>
"""


     components.html(html_content, height=300)

     col1, col2, col3 = st.columns(3)

     with col1:
          st.write(' ')

     with col2:
          # Record audio
          audio_data = audio_recorder()

          # If audio is recorded, display the audio player
          if audio_data:
             st.audio(audio_data, format="audio/wav")

          # Save the audio file
          with open("recorded_audio.wav", "wb") as f:
               f.write(audio_data)

     with col3:
          st.write(' ')

     audio_file_path = "recorded_audio.wav"

     if os.path.exists(audio_file_path):
        if st.button("Submit Recording"):
           audio_file= open("recorded_audio.wav", "rb")
           transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
           st.write("### Transcription :")
           st.write(transcription.text)
           dataframed = pd.read_csv('Dataset/ChatGPT_Prompt_Engineering_Cert_Basic_Modified.csv', encoding='ISO-8859-1')
           dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
           documents = dataframed['combined'].tolist()
           System_Prompt = """
You are an ai assistant that will take in a multiple choice questions alongside the user's final answer and explanation.

The User's answer needs to provide the final answer alongside their explanation on why that is the answer. The grading of each question must be 0.7 for the final answer and 0.3 for the explanation. If the User is wrong, the ai will provide the correct answer and explanation on why it is the correct answer. Else if the user is correct, the AI will acknowledge it accordingly. The AI will evaluate the user's answer and provide the user with feedback.


To Score the Explanation of the User per question, you will be providing a general criteria in which will grant the user 0.1 of the 0.3 points for explanation if they are able to achieve atleast one of it.

General Criteria :
Accuracy of Information: The explanation should accurately reflect established knowledge about AI and machine learning, specifically regarding prompt engineering.

Clarity of Explanation: The explanation should be clear and understandable, avoiding technical jargon where possible or defining such terms when used.

Relevance to the Question: Ensure that the explanation directly addresses the question asked, linking back to the specific techniques or concepts referenced in the question.

Depth of Insight: Look for explanations that provide depth beyond a superficial understanding, indicating critical thinking or application of the concept.

Evidence of Practical Understanding: Favor explanations that include practical examples or real-world applications of the concept, demonstrating how it can be applied outside theoretical contexts.

Then for the 0.2 points of the 0.3 Points for explanation will be coming from the provided Explanation Criteria in the context for each question. A user who atleast fulfills one explanation criteria will be granted 0.1 points, while a user who fulfills atleast two explanation criteria will be granted 0.2 points.

Question :
What is a potential benefit of using ensemble techniques with ChatGPT models?
A. Increases model interpretability
B. Boosts response diversity
C. Reduces training time
D. Enhances language fluency

""" + "Explanation Criteria : \n" + dataframed['Explanation_Criteria'][9] + "\n" + "Correct Answer : " + dataframed['Correct Answer'][9]
           
           struct = [ {"role": "system", "content": System_Prompt} ]

           user_answer = "User Answer and Explanation : " + transcription.text
           struct.append({"role" : "user", "content" : user_answer})
           chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct, temperature=0.2, max_tokens=3500, top_p=1, frequency_penalty=0, presence_penalty=0)
           response = chat.choices[0].message.content
           st.write("Feedback : ")
           st.write(response)
     else:
           st.write("No recorded audio available. Please record audio first.")



     html_content = """
<div style="border: 2px solid #333; border-radius: 8px; padding: 20px; width: 400px; margin: 20px auto; background-color: #f9f9f9; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
    <div style="font-size: 24px, font-weight: bold; margin-bottom: 10px; color: #000;"> Question 6</div>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #000;"> What is a potential drawback of using large-scale data augmentation in training ChatGPT models?</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">A. Increased computational costs</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">B. Reduced model generalization</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">C. Lower response accuracy</div>
    <div style="font-size: 16px; margin-bottom: 5px; color: #000;">D. Limited training data availability</div>
</div>
"""


     components.html(html_content, height=300)

     col1, col2, col3 = st.columns(3)

     with col1:
          st.write(' ')

     with col2:
          # Record audio
          audio_data = audio_recorder()

          # If audio is recorded, display the audio player
          if audio_data:
             st.audio(audio_data, format="audio/wav")

          # Save the audio file
          with open("recorded_audio.wav", "wb") as f:
               f.write(audio_data)

     with col3:
          st.write(' ')

     audio_file_path = "recorded_audio.wav"

     if os.path.exists(audio_file_path):
        if st.button("Submit Recording"):
           audio_file= open("recorded_audio.wav", "rb")
           transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
           st.write("### Transcription :")
           st.write(transcription.text)
           dataframed = pd.read_csv('Dataset/ChatGPT_Prompt_Engineering_Cert_Basic_Modified.csv', encoding='ISO-8859-1')
           dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
           documents = dataframed['combined'].tolist()
           System_Prompt = """
You are an ai assistant that will take in a multiple choice questions alongside the user's final answer and explanation.

The User's answer needs to provide the final answer alongside their explanation on why that is the answer. The grading of each question must be 0.7 for the final answer and 0.3 for the explanation. If the User is wrong, the ai will provide the correct answer and explanation on why it is the correct answer. Else if the user is correct, the AI will acknowledge it accordingly. The AI will evaluate the user's answer and provide the user with feedback.


To Score the Explanation of the User per question, you will be providing a general criteria in which will grant the user 0.1 of the 0.3 points for explanation if they are able to achieve atleast one of it.

General Criteria :
Accuracy of Information: The explanation should accurately reflect established knowledge about AI and machine learning, specifically regarding prompt engineering.

Clarity of Explanation: The explanation should be clear and understandable, avoiding technical jargon where possible or defining such terms when used.

Relevance to the Question: Ensure that the explanation directly addresses the question asked, linking back to the specific techniques or concepts referenced in the question.

Depth of Insight: Look for explanations that provide depth beyond a superficial understanding, indicating critical thinking or application of the concept.

Evidence of Practical Understanding: Favor explanations that include practical examples or real-world applications of the concept, demonstrating how it can be applied outside theoretical contexts.

Then for the 0.2 points of the 0.3 Points for explanation will be coming from the provided Explanation Criteria in the context for each question. A user who atleast fulfills one explanation criteria will be granted 0.1 points, while a user who fulfills atleast two explanation criteria will be granted 0.2 points.

Question :
What is a potential drawback of using large-scale data augmentation in training ChatGPT models?
A. Increased computational costs
B. Reduced model generalization
C. Lower response accuracy
D. Limited training data availability

""" + "Explanation Criteria : \n" + dataframed['Explanation_Criteria'][5] + "\n" + "Correct Answer : " + dataframed['Correct Answer'][5]
           
           struct = [ {"role": "system", "content": System_Prompt} ]

           user_answer = "User Answer and Explanation : " + transcription.text
           struct.append({"role" : "user", "content" : user_answer})
           chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct, temperature=0.2, max_tokens=3500, top_p=1, frequency_penalty=0, presence_penalty=0)
           response = chat.choices[0].message.content
           st.write("Feedback : ")
           st.write(response)
     else:
           st.write("No recorded audio available. Please record audio first.")