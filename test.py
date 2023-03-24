while True:
    zimus = input("请输入不带数字的文本:")
    flag = False
    for zimu in zimus:
        if '0' <= zimu <= '9':
            flag = True
            continue
    if flag == False:
        break

print(len(s))
