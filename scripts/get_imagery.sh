# get and unpack imagery data
# run from root directory
wget -O data.zip http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip \
    && unzip data.zip \
    && rm data.zip \
    && mv UCMerced_LandUse/Images . \
    && mv Images data \
    && rm -r UCMerced_LandUse 