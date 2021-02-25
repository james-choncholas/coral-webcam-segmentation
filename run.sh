
# 15fps
#mymodel=bodypix_resnet_50_416_288_16_quant_edgetpu_decoder.tflite

# ?fps
#mymodel=bodypix_resnet_50_640_480_16_quant_edgetpu_decoder.tflite

# 5fps
#mymodel=bodypix_resnet_50_768_496_32_quant_edgetpu_decoder.tflite

#python3 bodypix.py \
#    --anonymize --nobodyparts \
#    --model models/$mymodel


#https://gstreamer.freedesktop.org/documentation/tutorials/basic/debugging-tools.html?gi-language=c
#export GST_DEBUG=4
export GST_DEBUG=2

#sudo modprobe v4l2loopback devices=1 video_nr=11 card_label="test" exclusive_caps=1 &
python3 bodypix.py
