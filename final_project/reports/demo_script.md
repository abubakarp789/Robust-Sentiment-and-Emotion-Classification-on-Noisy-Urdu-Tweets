# Streamlit Application Demo Script

This document provides a 2-3 minute step-by-step presentation script for demonstrating the **Leakage-Aware Urdu Tweet Sentiment Classification** Streamlit web application.

---

## Demo Overview
* **Target Audience**: Course instructors, peers, or grading committee.
* **Duration**: 2.5 minutes (approximately 150 seconds).
* **Key Focus**: Showing how clean preprocessing, emoji stripping, and explanation templates help address leakage and interpret model decisions.

---

## Timing & Script Walkthrough

### Section 1: Introduction (0:00 - 0:30)
* **Presenter Action**: Open the Streamlit web application in a browser. Share the screen. Point to the sidebar and header.
* **Visual**: Clean, premium dark/glassmorphic web interface showing the title, project information, ethics details, and the model comparison leaderboard.
* **Spoken Script**:
  > "Hello everyone. Today I'm demonstrating our interactive deployment application for Leakage-Aware Urdu Tweet Sentiment Classification. 
  > 
  > On the left sidebar, you can see key project context, including details on SentiUrdu-1M's severe class imbalance—where Positive tweets make up over 86% and Neutral tweets comprise less than 0.3%. 
  > 
  > Underneath, we display a live **Model Comparison Leaderboard** showing our experimental results across 7 classifiers. Note that our classical **TF-IDF + Linear SVM** baseline is highlighted as the top-performing model, achieving a Test Macro-F1 of `0.5040`."

---

### Section 2: Input and Leakage-Free Preprocessing (0:30 - 1:15)
* **Presenter Action**: Copy a sample Urdu tweet containing emojis (e.g. `آج کا دن بہت اچھا ہے! 😊❤️`) and paste it into the "Enter an Urdu tweet" text box. Select **TF-IDF + Linear SVM (Best Overall)** and click the **Classify Sentiment** button.
* **Visual**: The preprocessed text box shows the cleaned text with emojis removed.
* **Spoken Script**:
  > "Let's test the interface. I am pasting an Urdu tweet: *'آج کا دن بہت اچھا ہے! 😊'* which means *'Today is a very good day!'* along with happy emojis.
  > 
  > When I click 'Classify Sentiment', our pipeline immediately preprocesses the input. 
  > 
  > Notice under the 'Preprocessed Text' box that the emojis—the smiling face and red heart—have been stripped out. This is a critical design feature. Since the dataset labels were weakly supervised using emojis, leaving them in would create **label leakage**, causing models to learn emoji heuristics instead of the underlying Urdu text semantics. Our pipeline ensures leakage-free inference."

---

### Section 3: Classification & Explanation Assistant (1:15 - 2:00)
* **Presenter Action**: Highlight the prediction result, the probability distribution bar chart, and the Explanation Assistant text box.
* **Visual**: A green "Positive" label displays with a high confidence score (e.g., `0.9500`). The bar chart shows a high probability for the Positive class and negligible probabilities for others. The Explanation Assistant displays a text block explaining the prediction.
* **Spoken Script**:
  > "The model predicts the sentiment as **Positive** with very high confidence. Below this, a dynamic bar chart displays the class probability distribution, showing a clear skew toward Positive.
  > 
  > Underneath, our **Explanation Assistant**—a lightweight, rule-based diagnostic helper—explains the output. It reads the token length and notes that the text contains no negation keywords like 'نہیں'. 
  > 
  > In case of a misclassification, this assistant automatically references historical error profiles, helping developers debug whether the error stems from minority-class confusion, short text ambiguity, or a polarity swap."

---

### Section 4: Robust Model Fallback Logic (2:00 - 2:30)
* **Presenter Action**: Select a transformer model (e.g., **XLM-RoBERTa**) from the selectbox. Mention that transformer models can be computationally heavy or require separate local checkpoint files. Click **Classify Sentiment**.
* **Visual**: A warning message appears saying: `Could not load selected model (xlm_roberta). Falling back to the baseline Linear SVM model.`, followed by a success message indicating the fallback model was loaded and inference completed.
* **Spoken Script**:
  > "Our web app is designed for production robustness. Multilingual transformers like XLM-RoBERTa can be large and occasionally fail to load if local checkpoints are missing or if VRAM is exhausted. 
  > 
  > If I choose XLM-RoBERTa and click classify, the system attempts to load it, but gracefully falls back to our robust Linear SVM baseline if the transformer is unavailable. This prevents application crashes and guarantees continuous uptime.
  > 
  > This concludes our demo. The Streamlit app successfully bridges our research findings into an interactive, leakage-aware tool. Thank you."

---

## Appendix: Demo Sample Inputs

Here are three quick-copy samples to use during live grading or recording:

1. **Positive Tweet (contains emojis):**
   * **Raw Text**: `آج کا دن بہت اچھا اور خوبصورت گزرا! 🌸✨`
   * **Cleaned Text**: `آج کا دن بہت اچھا اور خوبصورت گزرا`
   * **Expected Sentiment**: Positive

2. **Negative Tweet (contains negation and punctuation):**
   * **Raw Text**: `مجھے یہ بالکل بھی پسند نہیں آیا، بہت برا تجربہ تھا۔ 😡`
   * **Cleaned Text**: `مجھے یہ بالکل بھی پسند نہیں آیا بہت برا تجربہ تھا`
   * **Expected Sentiment**: Negative
   * **Explanation Assistant Highlight**: Will flag the presence of negation word `نہیں`.

3. **Neutral / Short Tweet (scarcity class):**
   * **Raw Text**: `کیا آپ وہاں ہیں؟`
   * **Cleaned Text**: `کیا آپ وہاں ہیں`
   * **Expected Sentiment**: Neutral (or falls back to Positive/Negative depending on model bias)
   * **Explanation Assistant Highlight**: Will flag the tweet as short (<= 4 tokens), indicating low context and ambiguity.
