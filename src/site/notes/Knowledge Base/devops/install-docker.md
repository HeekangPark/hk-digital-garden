---
{"title":"Docker 설치하기","date_created":"2021-02-16","date_modified":"2023-01-23","tags":["docker","ubuntu"],"dg-publish":true,"alias":"Docker 설치하기","dg-path":"devops/install-docker.md","permalink":"/devops/install-docker/","dgPassFrontmatter":true,"created":"2021-02-16","updated":"2023-01-23"}
---


## 설치 전 확인

2023년 1월 현재 Docker가 공식적으로 지원하는 Ubuntu 버전은 다음과 같다. 이외의 버전에서는 정상 동작을 보장할 수 없다.

- Ubuntu Kinetic 22.10
- Ubuntu Jammy 22.04 (LTS)
- Ubuntu Focal 20.04 (LTS)
- Ubuntu Bionic 18.04 (LTS)

2023년 1월 현재 Docker가 공식적으로 지원하는 아키텍처는 다음과 같다. 이외의 아키텍처에서는 정상 동작을 보장할 수 없다.

- x86_64 (= amd64)
- armhf
- arm64
- s390x

만약 시스템에 구버전의 Docker가 설치되어 있다면 제거하는 것이 좋다.

```bash
sudo apt remove docker docker-engine docker.io containerd runc
```

만약 clean uninstallation을 하고 싶다면 다음 명령어를 실행한다.

```bash
sudo apt purge docker-ce docker-ce-cli containerd.io docker-compose-plugin docker-ce-rootless-extras
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd
```

아래 문서는 **Ubuntu Focal 20.04 (LTS), x86_64 환경에서 최신 버전의 Docker Engine(stable)를 설치**하는 상황을 가정하고 작성되었다. 만약 다른 환경에서의 설치법을 찾고 있다면 [Docker 공식 문서](https://docs.docker.com/engine/)를 참고하자.

## Docker Engine 설치하기

1. 설치시 필요한 패키지 설치

    설치에 필요한 패키지들을 설치한다.

    ```bash
    sudo apt update
    sudo apt -y install ca-certificates curl gnupg lsb-release
    ```

2. Docker 공식 GPG 키 추가

    Docker 공식 GPG 키를 시스템에 추가한다.

    ```bash
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    ```

3. Docker 저장소 추가

    Docker stable 버전의 저장소를 `apt`에 추가한다.

    ```bash
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    ```

4. Docker 설치

    최신 버전의 Docker Engine을 설치한다.

    ```bash
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    ```

5. 설치 검증

    다음 명령어를 실행하여 설치가 잘 되었는지 확인한다.

    ```bash
    sudo docker run hello-world
    ```

6. 사용자 사용 권한 추가

    이렇게 설치된 Docker의 모든 기능을 사용하려면 root 권한이 필요하다. 현재 사용자에 권한을 부여하여 `sudo` 없이도 Docker를 사용할 수 있도록 하자. 아래 명령어를 입력한 후 로그아웃했다가 다시 로그인하면 `sudo` 없이도 Docker를 사용할 수 있다.

    ```bash
    sudo usermod -aG docker $USER
    ```

## Docker Compose 설치하기

2023년 1월 현재 최신 Docker Compose는 v2.15.1이다. Docker Compose v1은 python으로 작성된 standalone한 구현체였던 반면, v2에서는 Docker의 확장이 되었다(구현 언어 golang). 우린 이미 위에서 `docker-compose-plugin`을 설치했기 때문에, (v1에서와는 다르게) 추가적으로 무언갈 더 설치할 필요 없다.

### command-line completion

Docker Compose가 v1에서 v2로 업그레이드되면서 이제 Docker Compose를 실행하기 위해서는 `docker-compose ***` 형태가 아닌 `docker compose ***`과 같은 형태로 써야 한다. backward-compatibility를 위해서 `docker-compose` 명령어도 사용 가능하긴 하나 `help` 등에서도 `docker compose`를 사용하여 설명하고 있는 것을 볼 수 있을 것이다.

그리고 그 여파로 Docker Compose의 command-line completion(자동완성) 기능이 사라졌다. 원래 Docker Compose v1에서는 bash, zsh에 대해 공식적으로 command-line completion 기능을 제공했으나, 버전업이 되면서 Docker의 sub command가 되어버렸기에 command-line completion 기능을 더 이상 제공하지 않게 된 것으로 보인다.

그런데 사실 `docker-compose`나 `docker compose`나 터미널 상에서 제공하는 명령어는 큰 차이가 없다. 따라서 command-line completion 기능이 정말 아쉽다면 v1의 것을 사용하는 방법이 있을 듯하다. bash의 경우 다음 명령어를 통해 Docker Compose v1.29.2의 command-line completion을 사용할 수 있다.

```bash
sudo curl -L https://raw.githubusercontent.com/docker/compose/1.29.2/contrib/completion/bash/docker-compose -o /etc/bash_completion.d/docker-compose # bash
```

단 v1의 command-line completion을 사용하면 (당연하겠지만) `docker-compose` 명령어에 대해서만 command-line completion을 사용할 수 있다. `docker compose` 명령어에 대해서는 여전히 command-line completion을 사용할 수 없다.