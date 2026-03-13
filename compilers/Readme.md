## Python3 Interpreters

- Here are my python3 manually compiled interpreter
- Which are compiled by myself
- Dependencies requires to build an python3.11.x interpreter are updated to newer version and doesn't works as expected to compile python3.11.x
- These are my pre-compiled parts of single interpreter. merge and install it by executing following snippet
  ``` bash
  sudo apt update
  sudo apt install make unzip
  cat Python3.11.12_a* > Python3.11.12.zip
  unzip Python3.11.12.zip
  cd Python-3.11.12
  sudo make install
  ```
- Thats it
