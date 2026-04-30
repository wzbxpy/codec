# CoDec

This is the code for [CoDec: Prefix-Shared Decoding Kernel for LLMs](https://arxiv.org/pdf/2505.17694)


## Environment

CUDA Toolkit 12.9

For CoDec On Ascend, the required hardware and software environment dependencies for this project are as follows:

- Ascend hardware:
	- [Atlas A2 Training / Inference Series Products](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)
	- [Atlas A3 Training / Inference Series Products](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)
	- Ascend 950PR/Ascend 950DT
- CPU architecture: `aarch64`/`x86_64`
- OS: Linux distributions supported by CANN, such as Ubuntu 20.04/22.04 and openEuler 22.03 SP4
- Software dependencies:
	- `gcc` >= 7.5, < 13.0
	- `cmake` >= 3.16
	- `python` >= 3.8, < 3.12
  - `CANN Toolkit` >= 8.5.0 (https://www.hiascend.com/cann).
- Recommended configurations:

| OS                             | `CANN` | `gcc` | `cmake` | `python` |
| ------------------------------ | ------ | ----- | ------- | -------- |
| Ubuntu 20.04.5                 | 8.5.0  | 9.3   | 3.16    | 3.10     |
| Ubuntu 22.04.5                 | 8.5.0  | 11.3  | 3.22    | 3.10     |
| openEuler 22.03 SP4            | 8.5.0  | 10.3  | 3.22    | 3.10     |


## Installation

```bash
uv pip install torch
uv pip install -Ue . --no-build-isolation
```

For Codec On Ascend, run the following build command in the project directory:

1. **Install the Community Edition CANN toolkit package**

Based on the category of [Ascend product](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html) you are using, download the corresponding CANN toolkit package `Ascend-cann-toolkit_{version}_linux-{arch}.run`. See [CANN toolkit](https://www.hiascend.com/zh/developer/download/community/result?module=cann) for the download link.

Then install the CANN toolkit package (for details, refer to the [CANN Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)).

```bash
# Ensure the installer has executable permission
chmod +x Ascend-cann-toolkit_{version}_linux-{arch}.run
# Install the CANN toolkit package
./Ascend-cann-toolkit_{version}_linux-{arch}.run --full --force --install-path=${install_path}
# Enable the CANN environment. For default path installation, taking root user as an example
# (for non-root users, replace /usr/local with ${install_path})
source /usr/local/Ascend/ascend_toolkit/set_env.sh
```
- `{version}`: CANN package version.
- `{arch}`: System architecture.
- `{install_path}`: Specified installation path, default is `/usr/local/Ascend`.

2. **Download and install dependencies**

Download the source code of this project, and execute the following commands in the project directory.

```bash
# Download project source code
git clone https://github.com/wzbxpy/codec.git
# Install the Python environment dependencies according to the requirements file.
```

```bash
# Build the specified example
cd catlass-faInfer-shared-prefix
bash scripts/build.sh flash_attention_infer_tla
```

If the following message appears, the build is successful.

```bash
"[INFO] Target "{flash_attention_infer_tla}" built successfully."
```

3. **Run the operator**

We have prepared the script for running and testing:
```bash
bash examples/flash_attention_infer_tla/run.sh
```
You can modify the following parameters in the script: 
```bash
batch, qSeqlen, kvSeqlen, numHeads, kvHeads, headSize, dtype="bf16"(or "half"), device, accCheck
```

## Evaluation

```bash
# kernel evaluation
scripts/kernel.sh

# end to end evaluation
scripts/e2e.sh
```
