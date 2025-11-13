#变量 函数 条件判断 循环
#导入库


def multiplication_table():
    """九九乘法表生成函数"""
    print("开始生成九九乘法表...")

    for row in range(1, 10):
        line = ""  # 初始化当前行字符串
        for col in range(1, row + 1):
            # 构建每个乘法表达式
            expression = f"{col}×{row}={col * row:2d}"
            line += expression + "  "  # 添加到当前行

        print(line)  # 打印完整的一行

    print("九九乘法表生成完成！")


# 调用函数
multiplication_table()

