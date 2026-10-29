# Viva Questions and Short Answers

1. **What is the aim of this project?**  
   To identify unusual electricity-consumption readings and flag them for human review.

2. **What is an anomaly?**  
   A data point or pattern that differs noticeably from expected behaviour.

3. **Does an anomaly prove electricity theft?**  
   No. It is only a screening signal; legitimate usage changes and meter faults can also cause anomalies.

4. **Why use Python?**  
   Python has useful libraries for data analysis, machine learning, visualization, and dashboards.

5. **What is Isolation Forest?**  
   An unsupervised algorithm that detects unusual points by isolating observations through random splits.

6. **Why is Isolation Forest suitable here?**  
   It can detect unusual patterns without requiring theft labels during training.

7. **What is feature engineering?**  
   Creating useful inputs from raw data, such as hour, weekday, rolling average, and deviation from average.

8. **Why clean the data?**  
   Missing timestamps, invalid values, and negative readings can distort analysis and model results.

9. **What is an anomaly score?**  
   A numeric indication of how unusual a reading appears to the model. In this project, higher values are more unusual.

10. **What is precision?**  
    True positives divided by all predicted positives.

11. **What is recall?**  
    True positives divided by all actual positives in the labelled test set.

12. **What is F1-score?**  
    The harmonic mean of precision and recall.

13. **What is a false positive?**  
    A normal-labelled reading that the model flags as unusual.

14. **What is a false negative?**  
    An injected anomalous reading that the model fails to flag.

15. **Why use a time-based split?**  
    To evaluate on later readings and reduce leakage from randomly mixing time periods.

16. **What does contamination mean?**  
    The approximate proportion of outliers assumed by Isolation Forest when choosing its decision threshold.

17. **Why is the dataset simulated?**  
    It makes the project reproducible and avoids requiring access to private utility data or hardware.

18. **What are the limitations?**  
    Synthetic patterns may not match real usage, and legitimate behaviour can be flagged.

19. **How can the project be improved?**  
    Validate on authorized real datasets, tune thresholds, compare models, and include human review.

20. **What is the role of Streamlit?**  
    It provides a simple interactive web dashboard for trends, flags, scores, and evaluation metrics.
