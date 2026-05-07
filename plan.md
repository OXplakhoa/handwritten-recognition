## Plan: MNIST CNN handwriting recognition

Build a report-forward, bilingual notebook workflow for MNIST handwritten digit recognition that satisfies the assignment requirements and supports a strong written explanation of the experiment. The recommended shape is a baseline CNN plus one tuned variant with mild data augmentation, evaluated using a held-out validation split and final test accuracy on the built-in MNIST test set.

**Steps**
1. Define the notebook structure and report narrative as one coordinated workflow: problem statement, dataset, preprocessing, baseline model, tuned model, evaluation, discussion, and conclusion. The notebook should produce all evidence needed for the written report.
2. Load MNIST from built-in Keras and document the dataset shape, class count, and fixed 60k/10k train-test split. Keep the test set untouched until final evaluation.
3. Apply minimal preprocessing for the baseline: normalize pixel values and convert labels to categorical form for multi-class classification.
4. Reserve an internal validation split from the 60k training set for model selection and comparison. Use this split only for tuning, not for final reporting.
5. Implement a baseline CNN that clearly uses Conv2D and MaxPooling layers, followed by a classifier head. Keep it intentionally simple so the report has a clean reference point.
6. Implement one tuned CNN variant that improves on the baseline with moderate, explainable changes and mild data augmentation. Use small rotation, zoom, and shift ranges only, so augmentation supports generalization without distorting MNIST digits too aggressively.
7. Train both models under the same evaluation protocol, capture training history, and compare validation performance. Use early stopping if it helps produce cleaner and more stable training curves.
8. Evaluate the chosen model once on the untouched MNIST test set and record final accuracy. Add a confusion matrix, training curves, and sample predictions so the report can discuss both overall performance and common mistakes.
9. Draft the report outline in bilingual form: Vietnamese narrative with English technical terms where useful. Cover dataset, preprocessing, architecture, augmentation choice, training setup, accuracy, visualizations, and conclusion/lessons learned.
10. Validate the notebook by rerunning the full pipeline from data loading through final test metrics, then inspect that the plots, accuracy values, and sample predictions are coherent and reproducible.
11. Create an interactive Gradio drawing demo as the final notebook section. The demo should install Gradio if not present, load the saved tuned model weights, provide a sketchpad canvas for users to draw a digit (0–9), preprocess the drawing to match MNIST format (grayscale, 28x28, normalized pixel values), and display the predicted digit label along with a confidence bar chart or probability distribution for all 10 classes. Launch the interface inline with `share=False` so it runs directly in the notebook. Ensure the demo is self-contained and works even if the notebook is rerun from scratch.

**Relevant files**
- Jupyter notebook to be created for the full workflow and experiments.
- A report document or outline to be created from the notebook results.

**Verification**
1. Run the notebook end-to-end and confirm that training completes without shape or label encoding errors.
2. Check that the baseline and tuned models both report validation metrics and that final test accuracy is measured only once on the untouched test split.
3. Confirm that the notebook outputs include training curves, confusion matrix, sample predictions, and evidence that augmentation is applied only in the tuned branch.
4. Review the notebook narrative and report outline to ensure they are bilingual, report-heavy, and aligned with the assignment requirements.
5. Test the Gradio demo by drawing a digit on the canvas and verifying that the model returns a prediction label and confidence scores for all 10 classes.
6. Confirm that the Gradio interface launches inline in the notebook with `share=False` and that it works correctly even when the notebook is rerun from a fresh kernel.

**Decisions**
- Deliverable shape: notebook plus report outline.
- Model strategy: baseline CNN plus one tuned variant.
- Augmentation scope: mild rotation, zoom, and shift only in the tuned variant.
- Data split: use train/validation/test, with validation carved from the 60k training set.
- Report language: bilingual.
- Success target: prioritize the highest reasonable MNIST accuracy rather than the smallest possible implementation.
- Report evidence: include accuracy plus training curves, confusion matrix, and sample predictions.
- Scope emphasis: report-forward, with the notebook designed to feed the report directly.

**Further Considerations**
1. If the team wants a stricter classroom-demo version, the tuned model can be kept small and the augmentation ranges can be reduced further.
2. If the report needs to be more concise, the notebook can still preserve the full experimental evidence while the write-up summarizes the baseline, tuned result, and final test accuracy.
