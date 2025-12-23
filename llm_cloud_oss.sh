#!/bin/bash

# ============================================================
# 云服务器工作目录 & OSS 备份管理脚本
# ============================================================
# 功能：管理临时工作目录和OSS持久化备份之间的同步
# 
# 使用方法：
#   bash llm_cloud_oss.sh init      # 初始化Python环境
#   bash llm_cloud_oss.sh backup    # 备份整个工作目录到OSS
#   bash llm_cloud_oss.sh restore   # 从OSS恢复（不覆盖已存在文件）
#   bash llm_cloud_oss.sh status    # 查看当前状态
# ============================================================

# ========== 配置区 ==========
ENV_NAME="llm_env"
WORK_DIR="/mnt/workspace/llmworks"
OSS_BACKUP_DIR="/mnt/workspace/my_oss_data/llmprojs"
VENV_PATH="$WORK_DIR/.venv"

# ========== 辅助函数 ==========
log() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

warn() {
    echo -e "\033[33m[WARN]\033[0m $1"
}

error() {
    echo -e "\033[31m[ERROR]\033[0m $1" >&2
    exit 1
}

success() {
    echo -e "\033[32m✓\033[0m $1"
}

# ========== 初始化环境 ==========
do_init() {
    echo ""
    log "========== 初始化环境 =========="
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        error "Python3 未安装"
    fi
    success "Python $(python3 --version 2>&1 | awk '{print $2}') 可用"
    
    # 确保工作目录存在
    mkdir -p "$WORK_DIR" || error "无法创建工作目录: $WORK_DIR"
    success "工作目录就绪: $WORK_DIR"
    
    # 提示是否需要恢复
    if [ -z "$(ls -A "$WORK_DIR" 2>/dev/null | grep -v '^\.')" ] && \
       [ -d "$OSS_BACKUP_DIR" ] && [ -n "$(ls -A "$OSS_BACKUP_DIR" 2>/dev/null)" ]; then
        warn "工作目录为空，但OSS中有备份"
        log "建议运行: bash $0 restore"
    fi
    
    # 创建或复用虚拟环境
    if [ -d "$VENV_PATH" ]; then
        log "虚拟环境已存在: $VENV_PATH"
    else
        log "创建虚拟环境: $VENV_PATH"
        python3 -m venv "$VENV_PATH" || error "创建虚拟环境失败"
        success "虚拟环境已创建"
    fi
    
    # 激活虚拟环境
    source "$VENV_PATH/bin/activate" || error "激活虚拟环境失败"
    
    # 升级pip
    log "升级 pip..."
    pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ -q 2>/dev/null || true
    
    # 安装ipykernel
    log "安装 ipykernel..."
    pip install ipykernel -i https://mirrors.aliyun.com/pypi/simple/ -q 2>/dev/null || true
    
    # 注册Jupyter内核
    if command -v jupyter &> /dev/null; then
        jupyter kernelspec list 2>/dev/null | grep -q "$ENV_NAME" && \
            jupyter kernelspec uninstall "$ENV_NAME" -y 2>/dev/null || true
        python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)" 2>/dev/null || true
    fi
    
    # 查找并安装所有项目的requirements.txt
    log "查找项目依赖..."
    local found=0
    while IFS= read -r req_file; do
        found=1
        log "安装依赖: $req_file"
        pip install -r "$req_file" -i https://mirrors.aliyun.com/pypi/simple/ 2>&1 | \
            grep -E "Successfully installed|Requirement already satisfied" || true
    done < <(find "$WORK_DIR" -maxdepth 3 -name "requirements.txt" -type f 2>/dev/null)
    
    [ $found -eq 0 ] && warn "未找到任何 requirements.txt"
    
    # 完成提示
    echo ""
    success "========== 环境初始化完成 =========="
    echo "   工作目录: $WORK_DIR"
    echo "   虚拟环境: $VENV_PATH"
    echo "   激活命令: source $VENV_PATH/bin/activate"
    echo ""
}

# ========== 备份到OSS ==========
do_backup() {
    echo ""
    log "========== 备份工作目录到 OSS =========="
    
    # 检查工作目录
    if [ ! -d "$WORK_DIR" ] || [ -z "$(ls -A "$WORK_DIR" 2>/dev/null)" ]; then
        error "工作目录为空: $WORK_DIR"
    fi
    
    # 检查是否有大目录（排除虚拟环境，它们永远不备份）
    log "检查工作目录..."
    local large_dirs=()
    for dir in "$WORK_DIR"/{model,output}; do
        if [ -d "$dir" ]; then
            local size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
            large_dirs+=("$(basename "$dir") ($size)")
        fi
    done
    
    # 如果有大目录，询问用户
    local skip_large=0
    if [ ${#large_dirs[@]} -gt 0 ]; then
        echo ""
        warn "发现大目录（可能导致备份很慢）："
        for item in "${large_dirs[@]}"; do
            echo "   - $item"
        done
        echo ""
        read -p "是否跳过这些大目录？[y/N] " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] && skip_large=1
    fi
    
    # 确保OSS目录存在（如果是文件则先删除）
    if [ -e "$OSS_BACKUP_DIR" ] && [ ! -d "$OSS_BACKUP_DIR" ]; then
        warn "OSS路径存在但不是目录，正在删除..."
        rm -f "$OSS_BACKUP_DIR"
    fi
    mkdir -p "$OSS_BACKUP_DIR" || error "无法创建OSS目录: $OSS_BACKUP_DIR"
    
    # 使用临时目录
    log "准备备份..."
    local temp_dir="$WORK_DIR/.backup_temp_$$"
    rm -rf "$temp_dir"
    mkdir -p "$temp_dir" || error "无法创建临时目录"
    
    # 固定排除的目录（可重新生成的）
    local always_exclude=(".venv" "venv" ".backup_temp" "__pycache__" ".ipynb_checkpoints" ".git")
    local optional_exclude=()
    [ $skip_large -eq 1 ] && optional_exclude=("model" "output")
    
    # 复制文件
    log "复制文件..."
    for item in "$WORK_DIR"/* "$WORK_DIR"/.[!.]*; do
        [ -e "$item" ] || continue
        local name=$(basename "$item")
        local skip=0
        
        # 检查是否需要排除
        for exclude in "${always_exclude[@]}" "${optional_exclude[@]}"; do
            if [[ "$name" == "$exclude"* ]]; then
                skip=1
                warn "跳过: $name"
                break
            fi
        done
        
        [ $skip -eq 0 ] && cp -r "$item" "$temp_dir/" 2>/dev/null || true
    done
    
    # 清理缓存文件
    find "$temp_dir" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$temp_dir" -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # 统计备份大小
    local backup_size=$(du -sh "$temp_dir" 2>/dev/null | awk '{print $1}')
    local file_count=$(find "$temp_dir" -type f 2>/dev/null | wc -l)
    log "待备份: $file_count 个文件, 约 $backup_size"
    
    # 同步到OSS
    log "同步到 OSS（可能需要几分钟）..."
    rm -rf "$OSS_BACKUP_DIR"/* 2>/dev/null || true
    
    local total_items=$(ls -1 "$temp_dir" 2>/dev/null | wc -l)
    local current=0
    
    for item in "$temp_dir"/*; do
        [ -e "$item" ] || continue
        current=$((current + 1))
        local name=$(basename "$item")
        echo -n "   [$current/$total_items] $name ... "
        cp -r "$item" "$OSS_BACKUP_DIR/" 2>/dev/null && echo "✓" || echo "✗"
    done
    
    # 清理临时目录
    rm -rf "$temp_dir"
    
    # 导出依赖列表
    if [ -d "$VENV_PATH" ]; then
        log "导出依赖列表..."
        source "$VENV_PATH/bin/activate" 2>/dev/null && \
        pip freeze > "$OSS_BACKUP_DIR/pip_freeze.txt" 2>/dev/null && \
        deactivate 2>/dev/null || warn "无法导出依赖列表"
    fi
    
    # 记录备份信息（使用echo逐行写入，兼容OSS文件系统）
    {
        echo "备份时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "备份来源: $WORK_DIR"
        echo "备份大小: $backup_size"
        echo "文件数量: $file_count"
        echo "Python版本: $(python3 --version 2>&1 | awk '{print $2}')"
        echo "排除: ${always_exclude[*]} ${optional_exclude[*]}"
    } > "$OSS_BACKUP_DIR/backup_info.txt" 2>/dev/null || warn "无法写入备份信息文件"
    
    # 完成提示
    echo ""
    success "========== 备份完成 =========="
    echo "   备份位置: $OSS_BACKUP_DIR"
    echo "   备份时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "   备份大小: $backup_size ($file_count 个文件)"
    echo ""
}

# ========== 从OSS恢复 ==========
do_restore() {
    echo ""
    log "========== 从 OSS 恢复工作目录 =========="
    
    # 检查OSS备份
    if [ ! -d "$OSS_BACKUP_DIR" ] || [ -z "$(ls -A "$OSS_BACKUP_DIR" 2>/dev/null | grep -v '^\.')" ]; then
        error "OSS备份为空: $OSS_BACKUP_DIR"
    fi
    
    # 显示备份信息
    if [ -f "$OSS_BACKUP_DIR/backup_info.txt" ]; then
        log "备份信息："
        cat "$OSS_BACKUP_DIR/backup_info.txt" | sed 's/^/   /'
    fi
    
    # 确保工作目录存在
    mkdir -p "$WORK_DIR" || error "无法创建工作目录: $WORK_DIR"
    
    # 恢复文件（不覆盖已存在的）
    log "恢复中（不覆盖已存在文件）..."
    for item in "$OSS_BACKUP_DIR"/*; do
        local name=$(basename "$item")
        # 跳过元数据文件
        [ "$name" = "backup_info.txt" ] || [ "$name" = "pip_freeze.txt" ] && continue
        # 只复制不存在的
        if [ ! -e "$WORK_DIR/$name" ]; then
            cp -r "$item" "$WORK_DIR/" 2>/dev/null || warn "复制 $name 失败"
        fi
    done
    
    # 完成提示
    echo ""
    success "========== 恢复完成 =========="
    echo "   恢复位置: $WORK_DIR"
    warn "   注意：已存在的文件未被覆盖"
    echo ""
    log "建议运行: bash $0 init"
    echo ""
}

# ========== 查看状态 ==========
do_status() {
    echo ""
    log "========== 当前状态 =========="
    echo ""
    
    # 工作目录
    echo "📁 工作目录: $WORK_DIR"
    if [ -d "$WORK_DIR" ] && [ -n "$(ls -A "$WORK_DIR" 2>/dev/null)" ]; then
        echo "   状态: ✓ 存在"
        echo "   内容:"
        ls -1 "$WORK_DIR" 2>/dev/null | grep -v "^\." | head -10 | sed 's/^/      - /' || echo "      (无)"
        [ $(ls -1 "$WORK_DIR" 2>/dev/null | grep -v "^\." | wc -l) -gt 10 ] && echo "      ..."
    else
        echo "   状态: ✗ 为空"
    fi
    echo ""
    
    # OSS备份
    echo "💾 OSS备份: $OSS_BACKUP_DIR"
    if [ -d "$OSS_BACKUP_DIR" ] && [ -n "$(ls -A "$OSS_BACKUP_DIR" 2>/dev/null | grep -v '^\.')" ]; then
        echo "   状态: ✓ 存在"
        [ -f "$OSS_BACKUP_DIR/backup_info.txt" ] && cat "$OSS_BACKUP_DIR/backup_info.txt" | sed 's/^/   /'
    else
        echo "   状态: ✗ 为空"
    fi
    echo ""
    
    # Python环境
    echo "🐍 Python环境: $VENV_PATH"
    if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/python" ]; then
        echo "   状态: ✓ 存在"
        echo "   版本: Python $("$VENV_PATH/bin/python" --version 2>&1 | awk '{print $2}')"
    else
        echo "   状态: ✗ 不存在"
    fi
    echo ""
    
    # 建议操作
    echo "💡 可用操作:"
    [ ! -d "$VENV_PATH" ] && echo "   - bash $0 init      # 初始化环境"
    [ -d "$WORK_DIR" ] && [ -n "$(ls -A "$WORK_DIR" 2>/dev/null)" ] && \
        echo "   - bash $0 backup    # 备份到OSS"
    [ -d "$OSS_BACKUP_DIR" ] && [ -n "$(ls -A "$OSS_BACKUP_DIR" 2>/dev/null)" ] && \
        echo "   - bash $0 restore   # 从OSS恢复"
    echo ""
}

# ========== 主入口 ==========
case "${1:-status}" in
    init)
        do_init
        ;;
    backup)
        do_backup
        ;;
    restore)
        do_restore
        ;;
    status)
        do_status
        ;;
    *)
        echo "用法: bash $0 {init|backup|restore|status}"
        exit 1
        ;;
esac
