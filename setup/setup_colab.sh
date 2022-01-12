#!/bin/bash

# pip install
python3 -m pip install -r ./setup/requirements_colab.txt

# Visual Studio Code :: Package list
pkglist=(
ms-python.python
tabnine.tabnine-vscode
njpwerner.autodocstring
kevinrose.vsc-python-indent
ms-ceintl.vscode-language-pack-ja
sbsnippets.pytorch-snippets
mosapride.zenkaku
)
for i in ${pkglist[@]}; do
  code --install-extension $i
done