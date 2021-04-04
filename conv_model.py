import torch
import numpy as np
import torchvision
import os
import argparse
parser = argparse.ArgumentParser()
arg_lists = []
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg
conv_arg = add_argument_group('conv args')
conv_arg.add_argument('--width', type=int, default=256)
conv_arg.add_argument('--height', type=int, default=256)
conv_arg.add_argument('--input', type=str, default='1.pth')
conv_arg.add_argument('--output_type', type=str, default='onnx')
conv_arg.add_argument('--output', type=str, default='out.onnx')
args, unparsed = parser.parse_known_args()

from model.cain import CAIN
print("Building model: CAIN")
model = CAIN(depth=3)
model = torch.nn.DataParallel(model).to("cuda")
checkpoint = torch.load(args.input)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()
model.eval()
model.half()
try:
    state_dict = model.module.state_dict()
except AttributeError:
    state_dict = model.state_dict()

torch.save(state_dict, 'temp_conv.pth')

del model
if args.output_type=="onnx":
    print("Building model: CAIN")
    model = CAIN(depth=3)
    checkpoint = torch.load("temp_conv.pth")
    model.load_state_dict(checkpoint)
    data = torch.randn((1, 3, int(args.height), int(args.width)))
    data1 = torch.randn((1, 3, int(args.height), int(args.width)))

    input_names = ["input_1", "input_2"]
    output_names = ["output_frame", "output_features"]
    torch.onnx.export(model, (data,data1), "converted.onnx", verbose=True, input_names=input_names, output_names=output_names)


if args.output_type=="torch2rt":
    from torch2trt import *
    print("Building model: CAIN")
    model = CAIN(depth=3)
    checkpoint = torch.load("temp_conv.pth")
    model.load_state_dict(checkpoint)
    model.cuda().half()
    data = torch.randn((1, 3, args.width, args.height)).cuda().half()
    data1 = torch.randn((1, 3, args.width, args.height)).cuda().half()
    model_trt = torch2trt(model.cuda(), [data,data1], fp16_mode=True)
    model_trt.half()
    torch.save(model_trt.state_dict(), 'converted.pth')


#os.system("python3 trt.py")
