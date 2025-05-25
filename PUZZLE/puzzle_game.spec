# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['puzzle_game.py'],
    pathex=[],
    binaries=[],
    datas=[('IMG_4052.jpg', '.')],
    hiddenimports=['PyQt5.QtWidgets', 'PyQt5.QtGui', 'PyQt5.QtCore'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='puzzle_game',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # sem terminal ao abrir
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='puzzle_game'
)
