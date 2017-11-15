from model import *

# select GPU, comment it if you are not going to use GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# filte out tensorflow log message
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main():
	model = CNN()

	model.train_cnn_with_png()

if __name__ == '__main__':
	main()