



# 

# 使用方法

## 在线Demo体验

请点击HuggingFace Demo链接: [(Hugging Face)🤗](https://huggingface.co/spaces/opendatalab/CDM-Demo)

## 本地安装CDM

CDM需要对公式进行渲染，需要相关依赖包，推荐在Linux系统安装配置

## 准备环境

需要的依赖包括：Nodejs, imagemagic, pdflatex，请按照下面的指令进行安装：

### 步骤.1 安装 nodejs

```
wget https://registry.npmmirror.com/-/binary/node/latest-v16.x/node-v16.13.1-linux-x64.tar.gz

tar -xvf node-v16.13.1-linux-x64.tar.gz

mv node-v16.13.1-linux-x64/* /usr/local/nodejs/

ln -s /usr/local/nodejs/bin/node /usr/local/bin

ln -s /usr/local/nodejs/bin/npm /usr/local/bin

node -v
```

### 步骤.2 安装 imagemagic

`apt-get`命令安装的imagemagic版本是6.x，我们需要安装7.x的，所以从源码编译安装：

（编译前需要确认系统内安装有libpng-dev，否则编译出来的magick无法支持cdm使用）
```
git clone https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.1

cd ImageMagick-7.1.1

./configure

make

sudo make install

sudo ldconfig /usr/local/lib

convert --version
```

### 步骤.3 安装 latexpdf

渲染中文公式需要中文字体，当前cdm中使用的是思源黑体Source Han Sans SC.

```
apt-get update

sudo apt-get install texlive-full
```

### step.4 安装 python 依赖

```
pip install -r requirements.txt
```


## 使用CDM

如果安装过程顺利，现在可以使用CDM对公式识别的结果进行评测了。

### 1. 批量评测 

- 准备输入的json文件

在UniMERNet上评测，可以用下面的脚本获取json文件:

```
python convert2cdm_format.py -i {UniMERNet predictions} -o {save path}
```

或者，也可以参考下面的格式自行准备json文件:

```
[
    {
        "img_id": "case_1",      # 非必须的key
        "gt": "y = 2z + 3x",
        "pred": "y = 2x + 3z"
    },
    {
        "img_id": "case_2",
        "gt": "y = x^2 + 1",
        "pred": "y = x^2 + 1"
    },
    ...
]
```

`注意在json文件中，一些特殊字符比如 "\" 需要进行转义, 比如 "\begin" 在json文件中就需要保存为 "\\begin".`

- 评测:

```
python evaluation.py -i {path_to_your_input_json}
```

### 2. 启动 gradio demo

```
python app.py
```