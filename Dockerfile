FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

RUN mkdir -p /workspace
RUN mkdir -p /workspace/output

WORKDIR /workspace
ADD requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

COPY src/ /workspace/src/

RUN chgrp -R 0 /workspace && chmod -R g+rwX /workspace

ENTRYPOINT ["torchrun", "/workspace/src/llama_recipes/finetuning.py"]
