---
{"title":"CUDA 이해하기","date_created":"2023-04-07","date_modified":"2023-04-07","tags":[],"dg-publish":true,"alias":"CUDA 이해하기","dg-path":"//devops/cuda.md","permalink":"///devops/cuda/","dgPassFrontmatter":true,"created":"","updated":""}
---


## CUDA란?

초창기의 컴퓨터에서는 CPU가 말 그대로 (유일한) '중앙처리장치'로서, 수치 계산 뿐만 아니라 모니터의 각 픽셀이 어떤 출력을 해야 하는지(= 얼마만큼 전기를 사용해야 하는지)까지 모두 다 계산했었다. 초창기에는 모니터의 화질(픽셀 수)도 별로 좋지 않고, 출력되는 내용도 텍스트나 아주 간단한 2D 그래픽 정도였기에 이게 가능했었다. 하지만 모니터의 화질이 점점 좋아지고, 다이나믹한 내용을 출력하고 싶어하는 수요가 점점 더 커지면서(ex. 게임) CPU만으로는 이 모든 연산을 다 처리할 수 없게 되었고, 이에 '화면 출력을 위한 연산'만을 전문적으로 처리하는 GPU가 등장한다.

한편 시간이 지나면서 컴퓨팅 환경은 텍스트 기반(CLI)에서 그래픽 기반(GUI)으로, 그리고 그 그래픽도 2차원 그래픽에서 3차원 그래픽으로 바뀌게 된다. 이때 3차원 그래픽 연산은 3차원 공간에 점을 찍고, 그 점들을 연결해 만들어진 선, 면들이 각 픽셀마다 어떤 색으로, 어떤 텍스쳐로, 어떤 광택을 가지고 보일지를 매 프레임마다 반복적으로 계산해야 하는 연산이다. 이런 종류의 연산을 위해서는 코어 하나하나의 처리 속도를 높이는 것보다, 코어 수를 무식하게 많이 때려박아 병렬 처리 능력을 높이는 것이 효율적이다. 이에 GPU는 점점 병렬 처리에 적합한 장치가 되어 갔다.

이때 이 GPU의 병렬 처리 능력을 화면 출력을 위한 연산에만 쓰는 것이 아니라 범용적인 (반복) 계산에 사용하자는, 이른바 GPGPU(General-Purpose computing on GPU)라는 아이디어가 나왔다. FP32 연산 기준 오늘날의 CPU는 잘 나와야 수 TFLOPS가 나오는 반면, GPU는 수십 TFLOPS를 가뿐히 찍으니 그 강력함은 짐작할 수 있으리라. 이에 NVIDIA는 자사의 GPU에서 GPGPU를 제대로, 편하게 사용할 수 있는 방법을 제공했으니 그것이 바로 CUDA(Compute Unified Device Architecture)이다. CUDA 외에도 OpenCL과 같은 GPGPU API가 있긴 하지만, GPGPU 분야에서는 CUDA가 사실상 표준에 가깝다.

## 구조

CUDA 플랫폼은 크게 NVIDIA Display Driver Package와 CUDA Toolkit, 이렇게 두 개의 컴포넌트로 구성된다.

![CUDA 구조](https://docs.nvidia.com/deploy/cuda-compatibility/graphics/CUDA-components.png)

- NVIDIA Display Driver Package
    OS에서 GPU와 통신할 수 있도록 하는 소프트웨어. CUDA Toolkit이 없더라도 이 driver만 설치되어 있다면 (GPU의 원래 목적인) 화면 출력은 할 수 있다.
- CUDA Toolkit
    GPGPU 기능을 사용하는 응용 프로그램을 작성하고 최적화할 수 있도록 하는 소프트웨어 개발 도구(라이브러리, 컴파일러, 디버거, 등등). CUDA Toolkit은 display driver보다 높은 수준의 추상화를 제공하기 때문에, 개발자들은 C, C++과 같이 본인들이 익숙한 프로그래밍 언어를 사용해 CUDA 응용 프로그램을 작성할 수 있다.

이처럼 driver와 CUDA Toolkit이 독립적인 계층으로 존재하기에, CUDA는 추상화되어 선택적으로 사용할 수 있는 기능이 되었다. CUDA Toolkit을 설치하지 않고 driver만 설치해 GPU를 오직 화면 출력용으로만 사용할 수도 있고, (선택적으로) CUDA Toolkit을 설치하면 응용 프로그램들에서 GPGPU 기능을 사용할 수 있게 되는 것이다. 그래서 CUDA를 '설치'한다는 말은 보통 CUDA Toolkit을 설치한다는 말로 사용된다.

## cuDNN이란?

cuDNN(CUDA Deep Neural Network library)은 일종의 CUDA 확장팩으로, DNN(Deep Neural Network)에서 일반적으로 사용되는 convolution, pooling, normalization 등과 같은 DNN 연산에 대해 고도로 최적화된 구현을 제공한다. TensorFlow, PyTorch 등의 딥러닝 프레임워크는 cuDNN을 사용하여 NVIDIA GPU에서의 학습/추론을 가속화할 수 있다.

## 버전 호환성

OS, display driver, CUDA Toolkit, cuDNN을 설치하려면 서로 버전을 잘 맞춰야 한다.

### OS - display driver

display driver는 설치 가능한 driver 버전 중 가장 높은 버전을 선택하는 것이 좋다.

Ubuntu의 경우 `ubuntu-drivers`를 사용하면 권장 driver를 추천받을 수 있다. 특별한 이유가 없다면 권장 driver를 설치하는 것이 좋다.

### display driver - CUDA Toolkit

CUDA Toolkit은 동작을 보장하기 위한 최소 요구 display driver 버전이 있다. 즉 display driver 버전이 높을 때 낮은 CUDA Toolkit 버전을 사용하는 것은 아무런 문제가 되지 않는다(호환성이 보장된다). 그러나 display driver 버전이 너무 낮다면 높은 버전의 CUDA Toolkit을 설치할 수 없다.

CUDA Toolkit 버전별 최소 요구 display driver 버전은 CUDA Toolkit release note를 참조하면 된다.

> CUDA Documentations : <https://docs.nvidia.com/cuda/archive/>

예를 들어 Linux x86_64에서 CUDA Toolkit 11.7.1 버전을 사용하고 싶다면, [release note](https://docs.nvidia.com/cuda/archive/11.7.1/cuda-toolkit-release-notes/index.html)를 참고, 450.80.02 버전 이상의 display driver를 사용하면 정상동작이 보장된다는 것을 확인할 수 있다.

### CUDA Toolkit - cuDNN

cuDNN은 특별한 이유가 없다면 CUDA Toolkit 버전에 맞춰 가장 높은 버전을 설치해주면 된다.

## 버전 확인

### GPU 확인

```bash title:"그래픽카드 모델명 확인"
lspci | grep -i VGA
```

### Display Driver 확인

```bash title:"nvidia-smi 이용해 display driver 버전 확인"
nvidia-smi
```

참고로 `nvidia-smi`에서 나오는 CUDA 버전은 현재 설치된 display driver가 지원하는 CUDA Toolkit 버전을 의미하는 것으로, 실제로 설치된 CUDA Toolkit 버전과 다를 수 있다. `nvidia-smi`로 나오는 CUDA 버전과 CUDA Toolkit 버전이 다르더라도, 상술했듯이 CUDA Toolkit의 최소 요구 display driver 버전만 맞으면 사용에 아무런 문제가 없다.

만약 `nvidia-smi`가 설치되어 있지 않다면 높은 확률로 display driver가 설치되어 있지 않은 것이다.

`apt`를 이용해 display driver를 설치했다면 다음 명령어로 설치된 display driver 버전을 확인할 수도 있다.

```bash title:"apt 이용해 display driver 버전 확인"
sudo apt list --installed | grep "nvidia"
```

### CUDA Toolkit 확인

```bash title:"CUDA Toolkit 버전 확인"
nvcc -V
```

## display driver, CUDA Toolkit, cuDNN 설치

NVIDIA RTX A6000 4장이 장착된 Ubuntu 22.04 시스템에 CUDA 11.7.1을 설치하는 시나리오를 생각해 보자.

1. 우선 그래픽카드가 올바르게 인식되고 있는지 확인한다.

    ```bash title:"그래픽카드 모델명 확인"
    lspci | grep -i VGA
    ```
    
2. display driver 설치

    `ubuntu-drivers` 명령어를 이용해 설치 가능한 driver 목록을 검색/설치한다.

    ```bash title:"설치 가능한 driver 목록 검색"
    sudo ubuntu-drivers devices
    ```

    검색된 driver는 `apt` 명령을 이용해 직접 설치해도 되고, 
    
    ```bash title:"driver 설치"
    sudo apt install nvidia-driver-xxx # xxx : driver 버전
    ```
  
   다음 명령어를 이용하면 권장 driver(`recommended`)를 바로 설치할 수 있다.

    ```bash title:"권장 driver 설치"
    sudo ubuntu-drivers autoinstall
    ```

3. CUDA Toolkit 설치

    <https://developer.nvidia.com/cuda-toolkit-archive>에서 CUDA Toolkit 버전(`CUDA Toolkit 11.7.1`), OS(`Linux`), Architecture(`x86_64`), Distribution(`Ubuntu`), Version(`22.04`)을 선택하고, runfile(local)을 선택하여 나온 명령어를 실행한다.

    ```bash title:"CUDA Toolkit 11.7.1 다운로드 및 설치"
    wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
    sudo sh cuda_11.7.1_515.65.01_linux.run
    ```

    ![CUDA Toolkit Installation 1](/img/user/KnowledgeBase/devops/cuda-toolkit-installation-1.png)
    
    다운로드한 CUDA Toolkit에는 display driver가 같이 포함되어 있다. 2번 과정에서 display driver를 이미 설치했기에, display driver를 설치할거면 기존 display driver를 지우고 진행하라는 안내가 뜰 것이다. 우린 CUDA Toolkit만 설치할 것이기 때문에 "Continue"를 선택하고 진행한다.

    ![CUDA Toolkit Installation 2](/img/user/KnowledgeBase/devops/cuda-toolkit-installation-2.png)

    다음은 최종 사용자 라이선스에 동의하라는 안내문이 뜨는데, "accept"를 입력하고 진행한다.

    ![CUDA Toolkit Installation 3](/img/user/KnowledgeBase/devops/cuda-toolkit-installation-3.png)

    설치할 항목을 선택하는 창이 뜬다. CUDA Toolkit을 제외한 나머지 항목은 모두 체크 해제한다. 완료했으면 "Install"을 선택해 설치를 진행한다.

    설치가 완료되었으면 `~/.bashrc` 또는 `~/.profile` 파일을 수정해 CUDA Toolkit이 설치된 경로를 `PATH`, `LD_LIBRARY_PATH` 환경변수에 추가해 준다. 참고로 아무런 추가 설정을 하지 않았을 때, CUDA Toolkit 11.7은 `/usr/local/cuda-11.7` 디렉토리에 설치된다.

    ```bash title:"CUDA Toolkit 경로 PATH, LD_LIBRARY_PATH에 추가"
    echo 'export CUDA_PATH="/usr/local/cuda-11.7"'
    echo 'export PATH="$CUDA_PATH/bin:$PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
    source ~/.bashrc
	```

    다음 명령어를 이용해 CUDA Toolkit 설치가 성공적으로 되었음을 검증한다.

    ```bash title:"CUDA Toolkit 버전 확인"
    nvcc -V
    ```

4. cuDNN 설치
 
    <https://developer.nvidia.com/rdp/cudnn-archive>에서 설치한 CUDA 버전과 호환되는 cuDNN 버전(ex. `cuDNN v8.8.1 for CUDA 11.x`)을 찾아 "Local Installer for Linux x86_64 (Tar)" 파일을 다운로드한다.

    다운로드가 완료되었으면 압축을 해제한다.

    ```bash title:"cuDNN tar.xz 압축 해제"
	tar -xvf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
	```
	
    다음 명령어들을 실행해 cuDNN을 설치한다.
    
	```bash title:"cuDNN 설치"
	cd cudnn-linux-x86_64-8.8.1.3_cuda11-archive
	sudo cp include/cudnn*.h /usr/local/cuda-11.7/include
	sudo cp lib/libcudnn* /usr/local/cuda-11.7/lib64
	sudo chmod a+r /usr/local/cuda-11.7/include/cudnn*.h /usr/local/cuda-11.7/lib64/libcudnn*
	```
	
    다음 명령어를 실행해 설치가 제대로 된 것을 확인한다.
    
	```bash title:"cuDNN 버전 확인"
	cat /usr/local/cuda-11.7/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
	```

### Troubleshooting

- 만약 `lspci` 명령어로 그래픽카드의 정확한 모델명(`GeForce RTX 3090`)이 안 나오고 `NVIDIA Corporation Device 2204`과 같은 형태로만 나오는 경우 다음 명령어를 입력해 PCI ID 리스트를 갱신한 후 `lspci` 명령어를 입력하면 된다.

    ```bash title:"PCI ID 리스트 갱신"
	sudo update-pciids
	```

- `ubuntu-drivers` 명령어가 없는 경우, `apt`를 이용해 설치한 후 명령어를 사용하면 된다.

    ```bash title:"ubuntu-drivers 설치"
    sudo apt install ubuntu-drivers
    ```

- CUDA Toolkit 설치를 완료했지만 `nvcc` 명령어가 없다고 나오는 경우, `PATH` 환경변수에 CUDA Toolkit 경로가 제대로 등록되지 않은 것이다. `PATH` 환경변수를 확인해보자.
