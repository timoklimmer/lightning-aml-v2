# update to latest available, use Azure ML Studio to identify latest
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-1.12-py39-cuda11.6-gpu:5

# adjust as needed (preferring concatenated commands to accelerate the image build process)
RUN pip install 'scikit-learn==1.2.0' \
                'opencv-contrib-python==4.7.0.68'
