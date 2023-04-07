import sys
import math
import numpy as np
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.resize(1000, 1000)
        self.glWidget = GLWidget()
        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.glWidget)
        self.setLayout(mainLayout)
        self.setWindowTitle("SPHERE")


class GLWidget(QOpenGLWidget):

    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        self.object = 0
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0

        self.lastPos = QPoint()

        self.main = QColor.fromCmykF(0.40, 0.0, 1.0, 0.0)
        self.clear = QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)

        img = cv2.imread("output/stitched pano.jpg")
        pgImData = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = pgImData
        self.mapWidth = self.img.shape[1]
        self.mapHeight = self.img.shape[0]
        self.inputMapFile = pgImData
        # self.inputMapFile = np.flipud(pgImData)

        self.view_scale = 0.5
        self.gl_width = 0
        self.gl_height = 0

    def sizeHint(self):
        return QSize(400, 400)

    def setXRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.xRot = angle
            self.update()

    def setYRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.yRot:
            self.yRot = angle
            self.update()

    def setZRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.zRot:
            self.zRot = angle
            self.update()

    # def setScale(self):


    def initializeGL(self):
        version_profile = QOpenGLVersionProfile()
        version_profile.setVersion(2, 0)
        self.gl = self.context().versionFunctions(version_profile)
        self.gl.initializeOpenGLFunctions()

        self.setClearColor(self.clear.darker())
        self.object = self.makeObject()
        self.gl.glShadeModel(self.gl.GL_SMOOTH)
        self.gl.glEnable(self.gl.GL_DEPTH_TEST)
        self.gl.glEnable(self.gl.GL_CULL_FACE)
        self.gl.glEnable(self.gl.GL_LIGHTING)
        self.gl.glLightModelfv(self.gl.GL_LIGHT_MODEL_AMBIENT, [0.9, 0.9, 0.9, 1.0])
        self.gl.glEnable(self.gl.GL_COLOR_MATERIAL)
        self.gl.glColorMaterial(self.gl.GL_FRONT, self.gl.GL_AMBIENT_AND_DIFFUSE)

        self.gl.glActiveTexture(self.gl.GL_TEXTURE0)
        self.text_obj = self.gl.glGenTextures(1)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.text_obj)
        self.gl.glPixelStorei(self.gl.GL_UNPACK_ALIGNMENT, 1)
        self.gl.glTexImage2D(self.gl.GL_TEXTURE_2D, 0, self.gl.GL_RGB, self.mapWidth, self.mapHeight, 0, self.gl.GL_RGB,
                             self.gl.GL_UNSIGNED_BYTE, self.inputMapFile.tobytes())
        self.gl.glPixelStorei(self.gl.GL_UNPACK_ALIGNMENT, 4)
        self.gl.glTexParameterf(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MAG_FILTER, self.gl.GL_LINEAR)
        self.gl.glTexParameterf(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MIN_FILTER, self.gl.GL_LINEAR)

    def paintGL(self):
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)
        self.gl.glLoadIdentity()
        self.gl.glTranslated(0.0, 0.0, -10.0)
        self.gl.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        self.gl.glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        self.gl.glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)

        self.gl.glEnable(self.gl.GL_TEXTURE_2D)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.text_obj)
        self.gl.glColor3f(1, 1, 1)
        self.gl.glCallList(self.object)

        width = self.gl_width
        height = self.gl_height
        self.gl.glMatrixMode(self.gl.GL_PROJECTION)
        self.gl.glLoadIdentity()
        if width <= height:
            self.gl.glOrtho(-self.view_scale, +self.view_scale, +self.view_scale * height / width,
                            -self.view_scale * height / width, 4.0, 15.0)
        else:
            self.gl.glOrtho(-self.view_scale * width / height, +self.view_scale * width / height,
                            +self.view_scale, -self.view_scale, 4.0, 15.0)
        self.gl.glMatrixMode(self.gl.GL_MODELVIEW)
        self.gl.glLoadIdentity()

    def resizeGL(self, width, height):
        side = min(width, height)
        self.gl_width = width
        self.gl_height = height
        if side < 0:
            return

        self.gl.glViewport(0, 0, width, height)
        self.gl.glMatrixMode(self.gl.GL_PROJECTION)
        self.gl.glLoadIdentity()
        if width <= height:
            self.gl.glOrtho(-self.view_scale, +self.view_scale, +self.view_scale * height / width,
                            -self.view_scale * height / width, 4.0, 15.0)
        else:
            self.gl.glOrtho(-self.view_scale * width / height, +self.view_scale * width / height,
                            +self.view_scale, -self.view_scale, 4.0, 15.0)
        self.gl.glMatrixMode(self.gl.GL_MODELVIEW)
        self.gl.glLoadIdentity()

    def wheelEvent(self, event):
        if self.view_scale - event.angleDelta().y() / 2400.0 > 0.7:
            self.view_scale = 0.7
        elif self.view_scale - event.angleDelta().y() / 2400.0 < 0.1:
            self.view_scale = 0.1
        else:
            self.view_scale -= event.angleDelta().y() / 2400.0
        self.update()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.setXRotation(self.xRot + 8 * dy)
            self.setYRotation(self.yRot + 8 * dx)
        elif event.buttons() & Qt.RightButton:
            self.setXRotation(self.xRot + 8 * dy)
            self.setZRotation(self.zRot + 8 * dx)

        self.lastPos = event.pos()

    def makeObject(self):
        genList = self.gl.glGenLists(1)
        self.gl.glNewList(genList, self.gl.GL_COMPILE)
        self.gl.glBegin(self.gl.GL_TRIANGLES)

        UResolution = 36
        VResolution = 36
        r = 1
        startU = 0
        startV = 0
        endU = math.pi * 2
        endV = math.pi
        stepU = (endU - startU) / UResolution  # step size between U-points on the grid
        stepV = (endV - startV) / VResolution  # step size between V-points on the grid
        for i in range(UResolution):  # U-points
            for j in range(VResolution):  # V-points
                u = i * stepU + startU
                v = j * stepV + startV
                un = endU if (i + 1 == UResolution) else (i + 1) * stepU + startU
                vn = endV if (j + 1 == VResolution) else (j + 1) * stepV + startV

                p0 = [math.cos(u) * math.sin(v) * r, math.cos(v) * r, math.sin(u) * math.sin(v) * r]
                p1 = [math.cos(u) * math.sin(vn) * r, math.cos(vn) * r, math.sin(u) * math.sin(vn) * r]
                p2 = [math.cos(un) * math.sin(v) * r, math.cos(v) * r, math.sin(un) * math.sin(v) * r]
                p3 = [math.cos(un) * math.sin(vn) * r, math.cos(vn) * r, math.sin(un) * math.sin(vn) * r]

                t0 = [i / UResolution, 1 - j / VResolution]
                t1 = [i / UResolution, 1 - (j + 1) / VResolution]
                t2 = [(i + 1) / UResolution, 1 - j / VResolution]
                t3 = [(i + 1) / UResolution, 1 - (j + 1) / VResolution]

                # Output the first triangle of this grid square
                self.gl.glTexCoord2f(*t0)
                self.gl.glVertex3f(*p0)
                self.gl.glTexCoord2f(*t2)
                self.gl.glVertex3f(*p2)
                self.gl.glTexCoord2f(*t1)
                self.gl.glVertex3f(*p1)

                # Output the other triangle of this grid square
                self.gl.glTexCoord2f(*t3)
                self.gl.glVertex3f(*p3)
                self.gl.glTexCoord2f(*t1)
                self.gl.glVertex3f(*p1)
                self.gl.glTexCoord2f(*t2)
                self.gl.glVertex3f(*p2)
                # t0 t2 t1 t3 t1 t2

        self.gl.glEnd()
        self.gl.glEndList()

        return genList

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

    def setClearColor(self, c):
        self.gl.glClearColor(c.redF(), c.greenF(), c.blueF(), c.alphaF())

    def setColor(self, c):
        self.gl.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())