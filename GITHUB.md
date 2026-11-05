# GitHub 同步到服务器

仓库：https://github.com/shuaijiji/ecommerce-retrieval （私有）

首次在 WebShell 中拉取到独立代码目录，再同步脚本到原项目：

```bash
cd /code
git clone https://github.com/shuaijiji/ecommerce-retrieval.git ecommerce-retrieval-source
cd /code/ecommerce-retrieval
mkdir -p backups
cp -a scripts "backups/scripts-$(date +%Y%m%d-%H%M%S)"
cp -a /code/ecommerce-retrieval-source/scripts/. scripts/
conda activate ecom-infer
find artifacts -type d -path '*/results/vision'
```

私有仓库需要服务器已有 GitHub 凭据。若 HTTPS 提示输入用户名和密码，用户名填 shuaijiji，密码位置使用有该仓库读取权限的访问令牌，不是 GitHub 登录密码。不要把令牌放入命令、仓库 URL 或聊天。也可以使用服务器已有的 SSH GitHub 访问配置。

后续更新：

```bash
git -C /code/ecommerce-retrieval-source pull --ff-only
cd /code/ecommerce-retrieval
cp -a scripts "backups/scripts-$(date +%Y%m%d-%H%M%S)"
cp -a /code/ecommerce-retrieval-source/scripts/. scripts/
```

数据、模型和结果继续保存在原项目目录。精排运行步骤见 scripts/RERANK.md；首次跑之前必须确认完整召回结果目录和 Reranker 模型已下载。上传代码不代表远端运行成功。
