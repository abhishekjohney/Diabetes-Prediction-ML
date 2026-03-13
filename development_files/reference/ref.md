
Automatic Zoom
Information 2025, 16, 7 22 of 31
Comparative Model Performance
Models like CatBoost and XGBoost, which also ranked high in our leaderboard,
achieved 83.06% validation accuracy. These models are well-suited for handling non-linear
relationships and imbalanced classes common in medical datasets. As observed in various
studies, diverse model architectures within ensemble frameworks enhance robustness,
capturing more complex data patterns [46–48].
5.4. Insights into Prediction Transparency with XAI
The integration of XAI techniques effectively addressed the “black-box” issue, provid-
ing insights into global feature importance and patient-specific predictions.
5.4.1. Global and Local Interpretability
SHAP analysis revealed glucose and BMI as the most critical predictors of diabetes
risk, quantitatively confirming their significance in the model’s decision-making. These
findings align with established clinical knowledge, where elevated glucose levels and
increased BMI are well-known risk factors for diabetes.
Beyond these primary predictors, SHAP provided insights into secondary features
such as the Diabetes Pedigree Function and Age, offering a more nuanced understanding
of individual patient risk. For example, the Diabetes Pedigree Function highlights genetic
predisposition, while Age reflects the increased likelihood of diabetes in older individuals.
These secondary insights enable clinicians to consider broader aspects of patient health
when evaluating risk.
By quantifying feature importance, SHAP not only validates the model’s alignment
with clinical standards but also provides actionable information. For instance, signifi-
cant contributions from glucose and BMI suggest prioritizing interventions like weight
management and glucose monitoring for at-risk individuals. Furthermore, the inclusion
of secondary predictors supports tailored care plans that address unique patient needs,
improving the effectiveness of diabetes management strategies.
5.4.2. LIME Analysis
LIME facilitated localized, patient-specific explanations by generating surrogate mod-
els to analyze individual predictions. For example, in a high-risk diabetes prediction, LIME
highlighted high glucose levels and elevated BMI as the primary contributors, consistent
with clinical thresholds for hyperglycemia (e.g., glucose > 140 mg/dL).
LIME’s case-specific insights enable healthcare professionals to validate individual
predictions, enhancing the reliability of the model in clinical contexts. Combined with
CA, LIME offers a powerful framework for exploring hypothetical scenarios. For instance,
clinicians can use CA to assess how reducing glucose levels might shift the risk classification,
providing actionable recommendations for lifestyle modifications.
5.4.3. Counterfactual Analysis
CA enables clinicians to explore how changes in input features could alter the model’s
predictions. By holding all other features constant, CA allows for simulations of hypotheti-
cal scenarios, such as reducing BMI or glucose levels, to evaluate their impact on diabetes
risk classification.
For example, a scenario involving a slight reduction in BMI from 35 to 30 demonstrated
a shift in the model’s prediction from high to low risk. This insight can guide clinicians in
recommending specific interventions, such as weight loss programs or dietary changes, to
mitigate diabetes risk.
Information 2025, 16, 7 23 of 31
When paired with LIME, CA becomes an invaluable tool for personalized care, en-
abling clinicians to tailor interventions based on individual patient data and simulate the
potential outcomes of these changes.
5.4.4. Quantifying Prediction Contributions with Integrated Gradients
IG provided a comprehensive view of feature attributions by measuring how the
model’s predictions change as input features transition from a baseline to their actual
values. This technique identified BMI and glucose as the most influential predictors,
corroborating findings from SHAP and LIME.
The IG analysis also revealed negative contributions from features such as Blood
Pressure, suggesting that lower blood pressure may slightly reduce the predicted diabetes
risk. By offering a detailed attribution for each feature, IG enhances transparency in high-
stakes clinical decisions, helping healthcare providers understand the factors driving the
model’s outputs.
5.4.5. Personalized Prediction Insights Using Attention Mechanism
AM assigns dynamic weights to features based on their relevance to individual pre-
dictions, complementing IG and SHAP analyses. AM consistently prioritized glucose and
BMI across predictions, reinforcing their critical role in diabetes risk assessment.
Unlike other techniques, AM adapts its focus based on the specific input data, allowing
for a tailored understanding of the model’s decision-making process. For instance, attention
weights provided by AM highlight the features most relevant to each patient, offering
additional context for clinicians when interpreting predictions.
By integrating AM with other XAI techniques, the framework ensures a holistic
view of the model’s behavior, combining dynamic feature prioritization with detailed
attributions. This robust interpretability makes the model a reliable and actionable tool for
healthcare professionals.
5.5. Comparison with Similar Studies
Although prior studies, such as [1,2], have successfully demonstrated the application
of XAI techniques like SHAP and LIME for diabetes prediction, our research advances the
field in several critical ways:
• Existing studies rely on manual model selection, which requires significant expertise
and may introduce bias in choosing algorithms. By integrating AutoML, our approach
automates model development, ensuring optimal performance across datasets while
reducing the technical barriers to implementing machine learning in healthcare.
• While SHAP and LIME provide robust global and local interpretability, CA adds a
new dimension by enabling clinicians to simulate how changes in specific features
(e.g., glucose or BMI) might alter outcomes. This capability supports personalized,
preventative care strategies, which are less explored in previous works.
• IG and the AM provide a holistic view of the model’s decision-making, making it easier
for clinicians to interpret and trust the model’s predictions. AM reveals the features
most emphasized by the model, while IG quantifies their individual contributions.
• Our Streamlit application bridges the gap between machine learning advancements
and clinical usability, offering a practical, accessible interface for healthcare profes-
sionals to interpret predictions and act on them in real-time. This aspect of our
work emphasizes the need for clinician-friendly tools, which is often missing in
theoretical studies.
These contributions address the limitations of prior works, which primarily focus on
demonstrating the applicability of XAI methods without incorporating the automation
Information 2025, 16, 7 24 of 31
and accessibility required for clinical adoption. While [3] effectively demonstrated the
use of counterfactuals for diabetes prevention through biomarker adjustments, their study
was limited to specific features within a single dataset. In contrast, our integration of CA
with AutoML ensures scalability across diverse datasets. Additionally, the incorporation
of SHAP and LIME enables clinicians to gain both global and local insights into model
predictions, addressing gaps in transparency and usability.
5.6. Methodological Innovations
This study introduces several methodological innovations that extend beyond standard
machine learning practices, making significant contributions to the field of healthcare AI:
• Traditional machine learning workflows rely heavily on manual model selection, which
can be time-consuming and require significant expertise. By automating the selection
and optimization process through AutoML, this study ensures robust performance
while democratizing access to advanced machine learning techniques.
• In addition to SHAP and LIME, which provide global and local interpretability, the
inclusion of CA offers a novel approach to understanding model predictions. By
allowing users to explore how minor adjustments in patient features affect outcomes,
CA supports individualized treatment planning, an aspect that is underexplored in
previous studies.
• Unlike purely theoretical approaches, this study bridges the gap between machine
learning and real-world healthcare applications by providing a user-friendly tool for
clinicians. The application integrates predictive insights and interpretability methods,
making it accessible and actionable for non-technical users.
These methodological innovations differentiate this research from standard workflows
and address critical challenges in healthcare AI, such as transparency, accessibility, and
clinical usability.
5.7. Practical Implications for Diabetes Prediction
Integrating AutoML with XAI introduces several practical benefits. AutoML simplifies
model selection and optimization, enabling healthcare professionals without extensive
machine learning expertise to implement predictive models with high accuracy. This aligns
with [10] findings that AutoML democratizes machine learning, allowing non-experts to
leverage AI in clinical applications.
5.7.1. Transparency for Clinical Decision-Making
The interpretability provided by SHAP and LIME promotes shared decision-making
between clinicians and patients. High SHAP values for BMI or Glucose can prompt
targeted lifestyle interventions, such as recommending weight loss programs or dietary
changes, while notable LIME contributions for age or family history may guide personalized
treatments, such as medication or tailored monitoring. This enables clinicians to prioritize
interventions based on the most influential features, ensuring that the treatment plan aligns
with the patient’s unique risk factors. Ref. [49] report that transparency in AI fosters patient
trust and improves clinical decision-making.
5.7.2. Actionable Insights for Personalized Care
SHAP and LIME provide clinicians with personalized, actionable insights that can
inform the patient’s care plan. For example, if the model indicates high risk due to elevated
BMI and Glucose levels, clinicians can utilize the SHAP values to inform patients about
the significance of controlling these factors. For instance, the model can suggest that
reducing BMI by 2–3 points could significantly reduce the risk of diabetes progression. This
Information 2025, 16, 7 25 of 31
actionable insight allows for personalized care and provides measurable goals for patients
to work toward, making the treatment plan concrete and realistic.
5.7.3. Using Counterfactual Analysis for Tailored Interventions
CA further enhances the model’s practical applicability by illustrating how small
changes in key features could influence predictions. For example, through CA, a clinician
could explore how changing a patient’s BMI or Glucose level might alter the model’s
prediction. If a patient is currently at high risk due to a BMI of 35, the clinician can show
the patient how a reduction in BMI by just 2–3 points could lower their risk. This feature
enables personalized decision-making, allowing clinicians to set tailored, realistic health
goals with clear visual feedback on how those changes could impact the patient’s future
health outcomes.
5.7.4. Shared Decision-Making: Enhancing Patient-Clinician Communication
These insights facilitate shared decision-making between clinicians and patients. By
using SHAP, LIME, and CA, clinicians can present actionable insights that help patients
understand the potential impacts of their lifestyle changes on their diabetes risk. This
collaborative approach strengthens patient engagement and ensures that patients are active
participants in shaping their care plans. The model’s ability to show real-time adjustments
to predictions based on lifestyle changes empowers both clinicians and patients to make
data-driven decisions together.
5.8. Limitations and Areas for Improvement
Despite its promising performance, this study has several limitations. The dataset’s
demographic specificity (Pima Indian population) may limit its generalizability to other
populations. Ref. [43] similarly noted that demographic limitations in machine learning
models could restrict broader applicability, recommending validation on larger, diverse
datasets. While this study uses the Pima Indian Diabetes dataset, which is a widely accepted
benchmark for diabetes prediction, the dataset primarily represents a specific ethnic group
and lacks diversity in features such as socioeconomic factors, genetic predispositions,
and lifestyle variables. To address this limitation, future work should involve validation
and retraining of the model on larger, more diverse datasets that include multi-ethnic
populations and varied geographical contexts. Additionally, the use of transfer learning
or federated learning could enhance generalizability by incorporating knowledge from
different datasets while maintaining data privacy.
5.8.1. Limitations in Interpretability Techniques
Although SHAP and LIME improve interpretability, they may not capture complex
feature interactions comprehensively. Ref. [50] recommend combining multiple inter-
pretability methods to provide deeper insights into model predictions. Furthermore, while
specificity is reasonable, reducing the false positive rate could improve the model’s clinical
applicability by minimizing unnecessary follow-ups and overdiagnosis.
5.8.2. Model Configuration Constraints
The relatively short AutoGluon training time may have limited the configurations
explored. Extending this training time in future experiments could allow for more robust
optimization, as other studies have found that prolonged training can enhance model per-
formance. This constraint also limits the exploration of more complex ensemble techniques,
which could yield additional performance improvements.
Information 2025, 16, 7 26 of 31
5.8.3. Future Directions
To enhance the generalizability of our findings, future work will focus on validating
the proposed model on diverse datasets, including multi-ethnic and geographically var-
ied populations. Collaborations with healthcare institutions to access real-world clinical
data can help ensure the model’s applicability to broader patient groups. Additionally,
incorporating transfer learning or federated learning approaches could enable the model
to learn from disparate datasets without compromising data privacy, thereby improving
both performance and generalizability. Further, enhancing the model’s architecture with
advanced AutoML capabilities and integrating deep learning algorithms into the AutoML
pipeline may yield higher predictive accuracy while maintaining interpretability. Finally,
exploring the impact of additional features, such as socioeconomic and environmental
factors, could further refine the model’s predictions and broaden its clinical relevance.
5.9. Key Contributions
This study makes several important contributions to the field of healthcare AI, particu-
larly in the context of diabetes prediction. By integrating AutoML with XAI techniques, we
provide a robust solution that balances predictive accuracy with interpretability, making
it both effective and usable in clinical settings. The following are the key contributions of
this research:
• Unlike previous studies that focus solely on either predictive accuracy or interpretabil-
ity, this research simultaneously addresses both challenges by combining AutoML
with XAI techniques like SHAP, LIME, and CA. This integration improves model
transparency while maintaining high prediction accuracy, which is essential for clin-
ical adoption. Our approach demonstrates that AutoML can not only automate the
model development process but also produce interpretable models, a crucial aspect
for healthcare applications.
• A significant contribution of this work is the development of a Streamlit-based ap-
plication, which allows clinicians to interact with the model, explore predictions,
and interpret the importance of different features in real time. This tool bridges the
gap between advanced machine learning techniques and real-world healthcare ap-
plications, making AI more accessible to healthcare professionals without machine
learning expertise.
• Our model demonstrates strong generalization capabilities, achieved through data
augmentation, feature engineering, and cross-validation. This robustness ensures
that the model performs consistently across diverse patient populations and datasets,
addressing a key limitation of many existing diabetes prediction models that struggle
with generalization.
• With SHAP and LIME, we provide clinically actionable insights into model predictions.
SHAP analysis offers global insights into feature importance, while LIME provides
localized, case-by-case explanations. This interpretability is essential for healthcare
professionals to make informed decisions based on model predictions and ensures the
AI system can be trusted in a clinical context.
• When compared to prior studies, our model achieves competitive performance while
addressing critical issues of transparency and interpretability. Table 1 compares the
accuracy and other evaluation metrics of our model with those of leading studies in
diabetes prediction. For example, while some studies like Tasin et al. [38] achieved
higher accuracy using XGBoost and SMOTE techniques, they did not provide the
same level of interpretability through XAI methods. In contrast, our study prioritizes
both performance and transparency, ensuring that the AI model can be reliably used
in clinical settings without sacrificing accuracy.
Information 2025, 16, 7 27 of 31
• By comparing the results of our model with those from other studies, we see that
while our accuracy 78.8% with generalization is competitive, our key contribution
lies in the combination of predictive accuracy and interpretability. Prior studies, such
as [15,38], achieved high accuracy but lacked the level of transparency that our model
offers through XAI methods like SHAP and LIME. This dual focus on performance
and interpretability is what sets our work apart and advances the field.
6. Conclusions
This study demonstrates the effectiveness of integrating AutoML with XAI techniques—
specifically SHAP, LIME, and CA—to enhance diabetes risk prediction. Using the Pima
Indian Diabetes dataset, the model achieved an accuracy of 76.62% on the primary test
set, which is consistent with recent benchmarks for diabetes prediction. Additionally, the
model demonstrated an average accuracy of 85.01% across multiple datasets, highlighting
its generalizability across diverse populations. While achieving competitive accuracy, the
focus of this study was on ensuring that the model is both interpretable and practical for
real-world healthcare applications.
The integration of SHAP and LIME enhances the model’s interpretability, aligning its
predictions with clinical reasoning. These tools provide actionable insights by quantifying
global feature importance and offering localized, patient-specific explanations. Further-
more, CA complements these techniques by allowing clinicians to explore how slight
adjustments in key features, such as BMI or Glucose, can influence individual predictions,
supporting personalized interventions.
To further facilitate clinical adoption, an interactive Streamlit application was de-
veloped. This application enables healthcare professionals to visualize patient-specific
risk factors, understand model behavior, and make informed, data-driven decisions. The
interface ensures that the model is not only accurate but also accessible and practical for
use in real-world clinical settings.
The model’s performance metrics, including sensitivity of 78.8% and specificity of
69.1%, demonstrate its balanced ability to detect diabetic cases and minimize unneces-
sary follow-ups. These attributes reinforce the model’s clinical applicability, validated
through robust cross-validation, and confirm its reliability across diverse patient popula-
tions. Among the ensemble methods tested, LightGBM Level 2 and Weighted Ensemble
Level 3 emerged as the most effective, showcasing the advantages of combining multiple
model architectures to capture complex patterns in health data.
While the accuracy achieved in this study is competitive, the primary focus remained
on developing a transparent and interpretable model suitable for real-world clinical appli-
cations. The integration of AutoML and XAI overcomes significant barriers to AI adoption
in healthcare by ensuring the model is not only accurate but also interpretable and easy
to use.
This research introduces several key innovations. AutoML simplifies model selection
and optimization, making advanced machine learning accessible to non-experts while
ensuring robust performance across datasets. The combination of SHAP, LIME, and CA
provides clinicians with personalized, actionable insights, addressing the common “black-
box” challenge of machine learning models. The Streamlit tool bridges the gap between AI
research and practical healthcare applications, offering an intuitive interface for clinicians
to interact with the model and make informed decisions.
Unlike prior works, which often focused on manual model selection or limited inter-
pretability tools, this study integrates automation and explainability into a single pipeline.
These contributions enhance the model’s accessibility, transparency, and trustworthiness,
making it suitable for clinical environments.
Information 2025, 16, 7 28 of 31
Despite its promising results, this study has limitations. The Pima Indian Diabetes
dataset primarily represents a specific ethnic group, which may limit the model’s generaliz-
ability. Future work should validate the model on larger, multi-ethnic datasets to ensure
fairness and applicability across diverse populations. While SHAP and LIME provide
robust insights, advanced techniques are needed to capture complex feature interactions,
particularly non-linear relationships, which could further improve transparency. Addition-
ally, while the model’s sensitivity is strong, reducing the false positive rate could further
enhance its clinical relevance by minimizing unnecessary follow-ups and overdiagnosis.
Future research will focus on validating this model on more diverse datasets, incor-
porating additional features such as socioeconomic and lifestyle factors, and exploring
advanced interpretability techniques. Integrating transfer learning or federated learning
approaches could also improve performance while maintaining data privacy. Additionally,
further optimization of the AutoML pipeline and model architecture may yield even higher
predictive accuracy while preserving interpretability.
This study advances the development of diabetes prediction models by integrating Au-
toML with XAI techniques, ensuring both high accuracy and interpretability. The proposed
model provides actionable insights and supports data-driven decision-making in clinical
settings. By offering transparency and ease of use, this approach fosters trust in AI applica-
tions, making it a valuable tool for healthcare professionals. The work lays the foundation
for future advancements in explainable and generalizable AI models in healthcare.
Author Contributions: R.H. and V.D. conceptualized and designed the overall research framework.
S.M. developed the experimental approach and provided essential materials. V.D. created and
implemented computational tools. R.H., V.D., S.M. and S.H. verified the accuracy and reliability
of the results. V.D. performed the statistical analyses, while S.M. conducted the experiments and
collected the data. V.D. organized and managed the research data. S.M. drafted the initial manuscript,
with R.H., V.D. and S.H. providing critical revisions and improvements. S.M. also prepared the
figures and visual representations. R.H. supervised the research process, overseeing each stage, and
secured funding for the project. V.D. coordinated project logistics. All authors have read and agreed
to the published version of the manuscript.
Funding: This research received no external funding.
Institutional Review Board Statement: Not applicable.
Informed Consent Statement: Not applicable.
Data Availability Statement: The datasets used in this study are available for download as follows: the
Pima Indians Diabetes Database, accessible on Kaggle (https://www.kaggle.com/datasets/uciml/pima-
indians-diabetes-database, 15 December 2024); the Scikit-learn Built-in Diabetes Dataset, utilized for
generalization purposes; and the Second Generalization Dataset (Rural African-American Patients), an
additional diabetes dataset from Kaggle (https://www.kaggle.com/datasets/imtkaggleteam/diabetes,
15 December 2024). The latter represents rural African-American patients and provides further insight
into the model’s ability to generalize across diverse demographic groups.
Acknowledgments: The authors would like to acknowledge the use of ChatGPT 24 May 2023
version (OpenAI, San Francisco, CA, USA), specifically to assist in some content for improved clarity
and effectiveness.
Conflicts of Interest: The authors declare no conflicts of interest.
List of Abbreviations
Information 2025, 16, 7 29 of 31
AI Artificial Intelligence
AutoML Automated Machine Learning
BMI Body Mass Index
F1-Score F1 Score (Harmonic Mean of Precision and Recall)
MCC Matthews Correlation Coefficient
ML Machine Learning
ROC-AUC Receiver Operating Characteristic—Area Under Curve
SHAP SHapley Additive exPlanations
XAI Explainable Artificial Intelligence
SVM Support Vector Machine
GDPR General Data Protection Regulation
LDL Low-Density Lipoproteins
HDL High-Density Lipoproteins
LIME Local Interpretable Model-Agnostic Explanations
References
1. Jakka, A.; Vakula Rani, J. An Explainable AI Approach for Diabetes Prediction. Innov. Comput. Sci. Eng. 2023, 565, 15–25.
[CrossRef]
2. Zhao, Y.; Chaw, J.K.; Ang, M.C.; Daud, M.M.; Liu, L. A Diabetes Prediction Model with Visualized Explainable Artificial
Intelligence (XAI) Technology. Adv. Vis. Inform. 2023, 14322, 648–661. [CrossRef]
3. Lenatti, M.; Carlevaro, A.; Guergachi, A.; Keshavjee, K.; Mongelli, M.; Paglialonga, A. A novel method to derive personalized
minimum viable recommendations for type 2 diabetes prevention based on counterfactual explanations. PLoS ONE 2022, 17,
e0272825. [CrossRef] [PubMed]
4. Waring, J.; Lindvall, C.; Umeton, R. Automated machine learning: Review of the state-of-the-art and opportunities for healthcare.
Artif. Intell. Med. 2020, 104, 101822. [CrossRef] [PubMed]
5. van der Schaar, M. AutoML and Interpretability: Powering the Machine Learning Revolution in Healthcare. In Proceedings of the
2020 ACM-IMS on Foundations of Data Science Conference, Virtual, 19–20 October 2020. [CrossRef]
6. Mustafa, A.; Rahimi Azghadi, M. Automated Machine Learning for Healthcare and Clinical Notes Analysis. Computers 2021,
10, 24. [CrossRef]
7. Thirunavukarasu, A.J.; Elangovan, K.; Gutierrez, L.; Li, Y.; Tan, I.; Keane, P.A.; Korot, E.; Ting, D.S.W. Democratizing Artificial
Intelligence Imaging Analysis With Automated Machine Learning: Tutorial. J. Med. Internet Res. 2023, 25, e49949. [CrossRef]
[PubMed]
8. Kavakiotis, I.; Tsave, O.; Salifoglou, A.; Maglaveras, N.; Vlahavas, I.; Chouvarda, I. Machine Learning and Data Mining Methods
in Diabetes Research. Comput. Struct. Biotechnol. J. 2017, 15, 104–116. [CrossRef] [PubMed]
9. Olisah, C.C.; Smith, L.; Smith, M. Diabetes mellitus prediction and diagnosis from a data preprocessing and machine learning
perspective. Comput. Methods Programs Biomed. 2022, 220, 106773. [CrossRef] [PubMed]
10. Ahmed Hashim, A.; Hameed Mousa, A. An evaluation framework for diabetes prediction techniques using machine learning.
BIO Web Conf. 2024, 97, 125. [CrossRef]
11. Duckworth, C.; Guy, M.J.; Kumaran, A.; O’Kane, A.A.; Ayobi, A.; Chapman, A.; Marshall, P.; Boniface, M. Explainable Machine
Learning for Real-Time Hypoglycemia and Hyperglycemia Prediction and Personalized Control Recommendations. J. Diabetes
Sci. Technol. 2024, 18, 113–123. [CrossRef] [PubMed]
12. Dharmarathne, G.; Jayasinghe, T.N.; Bogahawaththa, M.; Meddage, D.P.P.; Rathnayake, U. A novel machine learning approach
for diagnosing diabetes with a self-explainable interface. Healthc. Anal. 2024, 5, 100301. [CrossRef]
13. Tigga, N.P.; Garg, S. Prediction of Type 2 Diabetes using Machine Learning Classification Methods. Procedia Comput. Sci. 2020,
167, 706–716. [CrossRef]
14. Kumari, V.A.; Chitra, R. Classification of Diabetes Disease Using Support Vector Machine. Int. J. Eng. Res. Appl. 2013, 3,
1797–1801.
15. Sisodia, D.; Sisodia, D.S. Prediction of Diabetes using Classification Algorithms. Procedia Comput. Sci. 2018, 132, 1578–1585.
[CrossRef]
16. Behera, M.K.; Chakravarty, S. Diabetic Retinopathy Image Classification Using Support Vector Machine. In Proceedings of the
2020 International Conference on Computer Science, Engineering and Applications (ICCSEA), Gunupur, India, 13–14 March 2020;
pp. 1–4. [CrossRef]
17. Wu, J.; Diao, Y.; Li, M.; Fang, Y.; Ma, D. A semi-supervised learning based method: Laplacian support vector machine used in
diabetes disease diagnosis. Interdiscip. Sci. Comput. Life Sci. 2009, 1, 151–155. [CrossRef]
Information 2025, 16, 7 30 of 31
18. Alghurair, N.I.; Mezher, M.A. A Survey Study Support Vector Machines and K-MEAN Algorithms for Diabetes Dataset. Acad. J.
Res. Sci. Publ. 2020, 2, 14–25.
19. Chang, V.; Bailey, J.; Xu, Q.A.; Sun, Z. Pima Indians diabetes mellitus classification based on machine learning (ML) algorithms.
Neural Comput. Applic. 2023, 35, 16157–16173. [CrossRef] [PubMed]
20. Guan, Y.; Tsai, C.J.; Zhang, S. Research on Diabetes Prediction Model of Pima Indian Females. In Proceedings of the 2023 4th
International Symposium on Artificial Intelligence for Medicine Science, Chengdu China, 20–22 October 2023; pp. 294–303.
[CrossRef]
21. Sangroya, A.; Anantaram, C.; Rawat, M.; Rastogi, M. Using Formal Concept Analysis to Explain Black Box Deep Learning
Classification Models. In Proceedings of the 7th International Workshop “What Can FCA do for Artificial Intelligence”?
Co-Located with International Joint Conference on Artificial Intelligence (IJCAI 2019), Macao, China, 10 August 2019.
22. Dagliati, A.; Marini, S.; Sacchi, L.; Cogni, G.; Teliti, M.; Tibollo, V.; De Cata, P.; Chiovato, L.; Bellazzi, R. Machine Learning
Methods to Predict Diabetes Complications. J. Diabetes Sci. Technol. 2018, 12, 295–302. [CrossRef] [PubMed]
23. Erickson, N.; Mueller, J.; Shirkov, A.; Zhang, H.; Larroy, P.; Li, M.; Smola, A. AutoGluon-Tabular: Robust and Accurate AutoML
for Structured Data. arXiv 2020, arXiv:2003.06505.
24. Joseph, V.R. Optimal ratio for data splitting. Stat. Anal. Data Min. 2022, 15, 531–538. [CrossRef]
25. Verdonck, T.; Baesens, B.; Óskarsdóttir, M.; van den Broucke, S. Special issue on feature engineering editorial. Mach. Learn. 2024,
113, 3917–3928. [CrossRef]
26. Shorten, C.; Taghi, M. Khoshgoftaar A survey on Image Data Augmentation for Deep Learning. J. Big Data 2019, 6, 60. [CrossRef]
27. Bey, R.; Goussault, R.; Grolleau, F.; Benchoufi, M.; Porcher, R. Fold-stratified cross-validation for unbiased and privacy-preserving
federated learning. J. Am. Med. Inform. Assoc. 2020, 27, 1244–1251. [CrossRef] [PubMed]
28. Shchur, O.; Turkmen, C.; Erickson, N.; Shen, H.; Shirkov, A.; Hu, T.; Wang, Y. AutoGluon-TimeSeries: AutoML for Probabilistic
Time Series Forecasting. arXiv 2023, arXiv:2308.05566.
29. Mathotaarachchi, K.V.; Hasan, R.; Mahmood, S. Advanced Machine Learning Techniques for Predictive Modeling of Property
Prices. Information 2024, 15, 295. [CrossRef]
30. Ejiyi, C.J.; Qin, Z.; Amos, J.; Ejiyi, M.B.; Nnani, A.; Ejiyi, T.U.; Agbesi, V.K.; Diokpo, C.; Okpara, C. A robust predictive diagnosis
model for diabetes mellitus using Shapley-incorporated machine learning algorithms. Healthc. Anal. 2023, 3, 100166. [CrossRef]
31. Ghosh, S.K.; Khandoker, A.H. Investigation on explainable machine learning models to predict chronic kidney diseases. Sci. Rep.
2024, 14, 3687. [CrossRef]
32. Verma, S.; Boonsanong, V.; Hoang, M.; Hines, K.; Dickerson, J.; Shah, C. Counterfactual Explanations and Algorithmic Recourses
for Machine Learning: A Review. ACM CSUR 2024, 56, 1–42. [CrossRef]
33. Wang, Y.; Zhang, T.; Guo, X.; Shen, Z. Gradient based Feature Attribution in Explainable AI: A Technical Review. arXiv 2024,
arXiv:2403.10415.
34. Yan, R.; Shang, Z.; Wang, Z.; Xu, W.; Zhao, Z.; Wang, S.; Chen, X. Challenges and Opportunities of XAI in Industrial Intelligent
Diagnosis:Priori-empowered. Ji Xie Gong Cheng Xue Bao 2024, 60, 1.
35. Powers, D.M.W. Evaluation: From precision, recall and f-measure to roc, informedness, markedness and correlation. arXiv 2020,
arXiv:2010.16061. [CrossRef]
36. Chicco, D.; Tötsch, N.; Jurman, G. The Matthews correlation coefficient (MCC) is more reliable than balanced accuracy, bookmaker
informedness, and markedness in two-class confusion matrix evaluation. BioData Min. 2021, 14, 13. [CrossRef] [PubMed]
37. Tasin, I.; Nabil, T.U.; Islam, S.; Khan, R. Diabetes prediction using machine learning and explainable AI techniques. Healthc.
Technol. Lett. 2023, 10, 1–10. [CrossRef]
38. Curia, F. Explainable and transparency machine learning approach to predict diabetes develop. Health Technol. 2023, 13, 769–780.
[CrossRef]
39. Tuppad, A.; Patil, S.D. Machine learning for diabetes clinical decision support: A review. Adv. Comp. Int. 2022, 2, 22. [CrossRef]
[PubMed]
40. Dewage, K.A.K.W.; Hasan, R.; Rehman, B.; Mahmood, S. Enhancing Brain Tumor Detection Through Custom Convolutional
Neural Networks and Interpretability-Driven Analysis. Information 2024, 15, 653. [CrossRef]
41. Ahmed, K.F.; Uz Zaman, M.S.; Peyal, H.I.; Hossain, A.; Rahman Ratul, M.T.; Abdal, M.N.; Islam, M.I. An Interpretable Framework
for Predicting Type 2 Diabetes using ML and Explainable AI. In Proceedings of the 2023 26th International Conference on
Computer and Information Technology (ICCIT), Cox’s Bazar, Bangladesh, 13–15 December 2023; pp. 1–6. [CrossRef]
42. Mahmud, S.M.H.; Hossin, M.A.; Ahmed, M.R.; Noori, S.R.H.; Sarkar, M.N.I. Machine Learning Based Unified Framework for Diabetes
Prediction; ACM: New York, NY, USA, 2018; pp. 46–50.
43. SumaLata, G.L.; Joshitha, C.; Kollati, M. Prediction of Diabetes Mellitus using Artificial Intelligence Techniques. Scalable Comput.
Pract. Exp. 2024, 25, 3200–3213. [CrossRef]
44. Larabi-Marie-Sainte, S.; Aburahmah, L.; Almohaini, R.; Saba, T. Current Techniques for Diabetes Prediction: Review and Case
Study. Appl. Sci. 2019, 9, 4604. [CrossRef]
Information 2025, 16, 7 31 of 31
45. Kibria, H.B.; Nahiduzzaman, M.; Goni, M.O.F.; Ahsan, M.; Haider, J. An Ensemble Approach for the Prediction of Diabetes
Mellitus Using a Soft Voting Classifier with an Explainable AI. Sensors 2022, 22, 7268. [CrossRef] [PubMed]
46. Vivek Khanna, V.; Chadaga, K.; Sampathila, N.; Prabhu, S.; Chadaga, P.R.; Bhat, D.; Swathi, K.S. Explainable artificial intelligence-
driven gestational diabetes mellitus prediction using clinical and laboratory markers. Cogent Eng. 2024, 11, 2330266. [CrossRef]
47. Singh, A.; Dhillon, A.; Kumar, N.; Hossain, M.S.; Muhammad, G.; Kumar, M. eDiaPredict: An Ensemble-based Framework for
Diabetes Prediction. ACM TOMM 2021, 17, 1–26. [CrossRef]
48. Tanim, S.A.; Aurnob, A.R.; Shrestha, T.E.; Emon, M.R.I.; Mridha, M.F.; Miah, M.S.U. Explainable deep learning for diabetes
diagnosis with DeepNetX2. Biomed. Signal Process. Control. 2025, 99, 106902. [CrossRef]
49. Hendawi, R.; Li, J.; Roy, S. A Mobile App That Addresses Interpretability Challenges in Machine Learning–Based Diabetes
Predictions: Survey-Based User Study. JMIR Form. Res. 2023, 7, e50328. [CrossRef] [PubMed]
50. Long, C.K.; Puri, V.; Solanki, V.K.; Jeanette Rincon Aponte, G. An Explainable AI-Enabled Framework for the Diabetes Classi-
fication. In Proceedings of the 2023 IEEE International Conference on Machine Learning and Applied Network Technologies
(ICMLANT), San Salvador, El Salvador, 14–15 December 2023; pp. 1–6. [CrossRef]
Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual
author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to
