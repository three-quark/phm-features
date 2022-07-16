import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="phm_feature",
  version="0.0.5",
  author="qin_hain_ing",
  author_email="2364839934@qq.com",
  description="pick feature for phm",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://gitee.com/ultrapower_cd_2364839934_admin/phm-feature",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)
