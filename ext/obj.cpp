// OBJCORE- A Obj Mesh Library by Yining Karl Li
// This file is part of OBJCORE, Coyright (c) 2012 Yining Karl Li
// Modified by Chin-Ying Li, 2021

#include <iostream>
#include <limits>

#include "obj.h"

#define EPSILON std::numeric_limits<double>::epsilon()

obj::obj():
  boundingbox(32, 0.0)
{
	vbosize = 0;
	nbosize = 0;
	cbosize = 0;
	ibosize = 0;
	top = 0;
	defaultColor = glm::vec3(0,0,0);
	maxminSet = false;
	xmax=0; xmin=0; ymax=0; ymin=0; zmax=0; zmin=0; 
	
}

obj::~obj(){
	for(int i=0; i<faceboxes.size(); i++){
		delete faceboxes[i];
	}
}

void obj::buildVBOs(){
	recenter();
  // std::vector<float> VBOvec;
  // std::vector<float> NBOvec;
  // std::vector<int> IBOvec;
  // std::vector<float> TBOvec;
	int index = 0;
	bool genNormals = false;

  if(faces.size()!=facenormals.size())
  {
		genNormals = true;
	}
  bool hasTexture = facetextures.size() > 0;
  for(int k = 0; k < faces.size(); k++)
  {
    if(isConvex(faces[k])==true)
    {
		//if(0==0){
      const std::vector<int>& face = faces.at(k);
      const glm::vec4& p0 = points.at(face[0]);
			
      for(int i=2; i < face.size(); i++){
        glm::vec4& p1 = points[face[i-1]];
        glm::vec4& p2 = points[face[i]];
        vbo_vec.push_back(p0[0]) ; vbo_vec.push_back(p0[1]); vbo_vec.push_back(p0[2]); //vbo_vec.push_back(1.0f);
        vbo_vec.push_back(p1[0]); vbo_vec.push_back(p1[1]); vbo_vec.push_back(p1[2]); //vbo_vec.push_back(1.0f);
        vbo_vec.push_back(p2[0]); vbo_vec.push_back(p2[1]); vbo_vec.push_back(p2[2]); //vbo_vec.push_back(1.0f);

				if(genNormals==false){
          const std::vector<int>& facenormal = facenormals.at(k);
          nbo_vec.push_back(normals[facenormal[0]][0]); nbo_vec.push_back(normals[facenormal[0]][1]); nbo_vec.push_back(normals[facenormal[0]][2]); //nbo_vec.push_back(0.0f);
          nbo_vec.push_back(normals[facenormal[i-1]][0]); nbo_vec.push_back(normals[facenormal[i-1]][1]); nbo_vec.push_back(normals[facenormal[i-1]][2]); //nbo_vec.push_back(0.0f);
          nbo_vec.push_back(normals[facenormal[i]][0]); nbo_vec.push_back(normals[facenormal[i]][1]); nbo_vec.push_back(normals[facenormal[i]][2]); //nbo_vec.push_back(0.0f);
				}else{
                    
					glm::vec3 a = glm::vec3(p1[0], p1[1], p1[2]) - glm::vec3(p0[0], p0[1], p0[2]);
					glm::vec3 b = glm::vec3(p2[0], p2[1], p2[2]) - glm::vec3(p0[0], p0[1], p0[2]);
					glm::vec3 n = glm::normalize(glm::cross(a,b));
          nbo_vec.push_back(n[0]); nbo_vec.push_back(n[1]); nbo_vec.push_back(n[2]); //nbo_vec.push_back(0.0f);
          nbo_vec.push_back(n[0]); nbo_vec.push_back(n[1]); nbo_vec.push_back(n[2]); //nbo_vec.push_back(0.0f);
          nbo_vec.push_back(n[0]); nbo_vec.push_back(n[1]); nbo_vec.push_back(n[2]); //nbo_vec.push_back(0.0f);
				}

        if (hasTexture) {
          const std::vector<int>& facetexture = facetextures.at(k);
          tbo_vec.push_back(texturecoords[facetexture[0]].x); tbo_vec.push_back(texturecoords[facetexture[0]].y);
          tbo_vec.push_back(texturecoords[facetexture[1]].x); tbo_vec.push_back(texturecoords[facetexture[1]].y);
          tbo_vec.push_back(texturecoords[facetexture[2]].x); tbo_vec.push_back(texturecoords[facetexture[2]].y);
        }

        ibo_vec.push_back(index+0); ibo_vec.push_back(index+1); ibo_vec.push_back(index+2);

				index=index+3;
			}
		}
 	}
	
  vbosize = (int)vbo_vec.size();
  nbosize = (int)nbo_vec.size();
  ibosize = (int)ibo_vec.size();
  tbosize = (int)tbo_vec.size();
  /*
  for(int i=0; i<vbo_vec.size(); i++){
    vbo[i] = vbo_vec[i];
	}
	for(int i=0; i<NBOvec.size(); i++){
		nbo[i] = NBOvec[i];
	}
	for(int i=0; i<IBOvec.size(); i++){
		ibo[i] = IBOvec[i];
	}
  for (int i = 0; i<TBOvec.size(); i++){
    tbo[i] = TBOvec[i];
  }
*/
	setColor(glm::vec3(.4,.4,.4));
}

void obj::compareMaxMin(float x, float y, float z){
  if(maxminSet==true)
  {
    if(x > xmax){ xmax = x; }
    if(x < xmin){ xmin = x; }
    if(y > ymax){ ymax = y; }
    if(y < ymin){ ymin = y; }
    if(z > zmax){ zmax = z; }
    if(z < zmin){ zmin = z; }
		//cout << "lol" << endl;
		top = ymax;
		boundingbox[0]  =  xmin; boundingbox[1]  =  ymin; boundingbox[2]  =  zmax; boundingbox[3]  = 1.0f;
		boundingbox[4]  =  xmin; boundingbox[5]  =  ymax; boundingbox[6]  =  zmax; boundingbox[7]  = 1.0f;
		boundingbox[8]  =  xmax; boundingbox[9]  =  ymax; boundingbox[10] =  zmax; boundingbox[11] = 1.0f;
		boundingbox[12] =  xmax; boundingbox[13] =  ymin; boundingbox[14] =  zmax; boundingbox[15] = 1.0f;
		boundingbox[16] =  xmin; boundingbox[17] =  ymin; boundingbox[18] =  zmin; boundingbox[19] = 1.0f;
		boundingbox[20] =  xmin; boundingbox[21] =  ymax; boundingbox[22] =  zmin; boundingbox[23] = 1.0f;
		boundingbox[24] =  xmax; boundingbox[25] =  ymax; boundingbox[26] =  zmin; boundingbox[27] = 1.0f;
		boundingbox[28] =  xmax; boundingbox[29] =  ymin; boundingbox[30] =  zmin; boundingbox[31] = 1.0f;
  }
  else
  {
		xmax = x; xmin = x; ymax = y; ymin = y; zmax = z; zmin = z;
		top = ymax;
		maxminSet = true;
	}
}

bool obj::isConvex(std::vector<int>& face)
{
  if(face.size()<=3)
  {
		return true;
	}

	int k = (int)face.size()-1;
	glm::vec3 a = glm::vec3(points[face[0]][0], points[face[0]][1], points[face[0]][2]) - glm::vec3(points[face[k]][0], points[face[k]][1], points[face[k]][2]);
	glm::vec3 b = glm::vec3(points[face[0]][0], points[face[0]][1], points[face[0]][2]) - glm::vec3(points[face[1]][0], points[face[1]][1], points[face[1]][2]);
	glm::vec3 n = glm::normalize(glm::cross(a,b));

  for(int i=2; i < face.size(); i++){
		glm::vec3 c = glm::vec3(points[face[i-1]][0], points[face[i-1]][1], points[face[i-1]][2]) - glm::vec3(points[face[i-2]][0], points[face[i-2]][1], points[face[i-2]][2]);
		glm::vec3 d = glm::vec3(points[face[i-1]][0], points[face[i-1]][1], points[face[i-1]][2]) - glm::vec3(points[face[i]][0], points[face[i]][1], points[face[i]][2]);
		glm::vec3 m = glm::normalize(glm::cross(c,d));

		if((abs(m[0] - n[0]) > EPSILON) || (abs(m[1] - n[1]) > EPSILON) || (abs(m[2] - n[2]) > EPSILON)){	
			return false;
		}
	}

	glm::vec3 c = glm::vec3(points[face[k]][0], points[face[k]][1], points[face[k]][2]) - glm::vec3(points[face[k-1]][0], points[face[k-1]][1], points[face[k-1]][2]);
	glm::vec3 d = glm::vec3(points[face[k]][0], points[face[k]][1], points[face[k]][2]) - glm::vec3(points[face[0]][0], points[face[0]][1], points[face[0]][2]);
	glm::vec3 m = glm::normalize(glm::cross(c,d));
	if((abs(m[0] - n[0]) > EPSILON) || (abs(m[1] - n[1]) > EPSILON) || (abs(m[2] - n[2]) > EPSILON)){	
		return false;
	}
	return true;
}

void obj::addPoint(glm::vec3 point){
	
    if(points.size()==0)
    {
        xmin = point[0]; xmax = point[0];
        ymin = point[1]; ymax = point[1];
        zmin = point[2]; zmax = point[2];
    }
    
    points.push_back(glm::vec4(point[0], point[1], point[2], 1));
    
	compareMaxMin(point[0], point[1], point[2]);
}

void obj::addFace(std::vector<int> face)
{
    faces.push_back(face);
    float facexmax = points[face[0]][0]; float faceymax = points[face[0]][1]; float facezmax = points[face[0]][2]; 
    float facexmin = points[face[0]][0]; float faceymin = points[face[0]][1]; float facezmin = points[face[0]][2]; 
    for(int i=0; i<face.size(); i++){
        if(points[face[i]][0]>facexmax){ facexmax = points[face[i]][0]; }
		if(points[face[i]][0]<facexmin){ facexmin = points[face[i]][0]; }
		if(points[face[i]][1]>faceymax){ faceymax = points[face[i]][1]; }
		if(points[face[i]][1]<faceymin){ faceymin = points[face[i]][1]; }
		if(points[face[i]][2]>facezmax){ facezmax = points[face[i]][2]; }
		if(points[face[i]][2]<facezmin){ facezmin = points[face[i]][2]; }
    }
    float* facebox = new float[32];
    facebox[0]  =  facexmin; facebox[1]  =  faceymin; facebox[2]  =  facezmax; facebox[3]  = 1.0f;
    facebox[4]  =  facexmin; facebox[5]  =  faceymax; facebox[6]  =  facezmax; facebox[7]  = 1.0f;
    facebox[8]  =  facexmax; facebox[9]  =  faceymax; facebox[10] =  facezmax; facebox[11] = 1.0f;
    facebox[12] =  facexmax; facebox[13] =  faceymin; facebox[14] =  facezmax; facebox[15] = 1.0f;
    facebox[16] =  facexmin; facebox[17] =  faceymin; facebox[18] =  facezmin; facebox[19] = 1.0f;
    facebox[20] =  facexmin; facebox[21] =  faceymax; facebox[22] =  facezmin; facebox[23] = 1.0f;
    facebox[24] =  facexmax; facebox[25] =  faceymax; facebox[26] =  facezmin; facebox[27] = 1.0f;
    facebox[28] =  facexmax; facebox[29] =  faceymin; facebox[30] =  facezmin; facebox[31] = 1.0f;
    faceboxes.push_back(facebox);
}

void obj::addFaceNormal(std::vector<int> facen)
{
	facenormals.push_back(facen);
}

void obj::addFaceTexture(std::vector<int> facet)
{
	facetextures.push_back(facet);
}

void obj::addNormal(glm::vec3 normal)
{
	normals.push_back(glm::vec4(normal[0], normal[1], normal[2], 1));
}

void obj::addTextureCoord(glm::vec3 texcoord){
	texturecoords.push_back(glm::vec4(texcoord[0], texcoord[1], texcoord[2], 1));
}

std::vector<float> obj::getBoundingBox(){
	return boundingbox;
}

float obj::getTop(){
	return top;
}

void obj::recenter(){
	glm::vec3 center = glm::vec3((xmax+xmin)/2,ymin,(zmax+zmin)/2);
	xmax=points[0][0]-center[0]; xmin=points[0][0]-center[0]; 
    ymax=points[0][1]-center[1]; ymin=points[0][1]-center[1]; 
    zmax=points[0][2]-center[2]; zmin=points[0][2]-center[2]; 
    top=0;
	for(int i=0; i<points.size(); i++){
		points[i][0]=points[i][0]-center[0];
		points[i][1]=points[i][1]-center[1];
		points[i][2]=points[i][2]-center[2];
		compareMaxMin(points[i][0], points[i][1], points[i][2]);
	}
    
    for(int i=0; i<faceboxes.size(); i++){
        
        // std::vector<int>& face = faces[i];
        const std::vector<int>& face = faces.at(i);

        float facexmax = points[face[0]][0]; float faceymax = points[face[0]][1]; float facezmax = points[face[0]][2]; 
        float facexmin = points[face[0]][0]; float faceymin = points[face[0]][1]; float facezmin = points[face[0]][2]; 
        
        for(int j=0; j<face.size(); j++){
            if(points[face[j]][0]>facexmax){ facexmax = points[face[j]][0]; }
            if(points[face[j]][0]<facexmin){ facexmin = points[face[j]][0]; }
            if(points[face[j]][1]>faceymax){ faceymax = points[face[j]][1]; }
            if(points[face[j]][1]<faceymin){ faceymin = points[face[j]][1]; }
            if(points[face[j]][2]>facezmax){ facezmax = points[face[j]][2]; }
            if(points[face[j]][2]<facezmin){ facezmin = points[face[j]][2]; }
        }
        faceboxes[i][0]  =  facexmin; faceboxes[i][1]  =  faceymin; faceboxes[i][2]  =  facezmax; faceboxes[i][3]  = 1.0f;
        faceboxes[i][4]  =  facexmin; faceboxes[i][5]  =  faceymax; faceboxes[i][6]  =  facezmax; faceboxes[i][7]  = 1.0f;
        faceboxes[i][8]  =  facexmax; faceboxes[i][9]  =  faceymax; faceboxes[i][10] =  facezmax; faceboxes[i][11] = 1.0f;
        faceboxes[i][12] =  facexmax; faceboxes[i][13] =  faceymin; faceboxes[i][14] =  facezmax; faceboxes[i][15] = 1.0f;
        faceboxes[i][16] =  facexmin; faceboxes[i][17] =  faceymin; faceboxes[i][18] =  facezmin; faceboxes[i][19] = 1.0f;
        faceboxes[i][20] =  facexmin; faceboxes[i][21] =  faceymax; faceboxes[i][22] =  facezmin; faceboxes[i][23] = 1.0f;
        faceboxes[i][24] =  facexmax; faceboxes[i][25] =  faceymax; faceboxes[i][26] =  facezmin; faceboxes[i][27] = 1.0f;
        faceboxes[i][28] =  facexmax; faceboxes[i][29] =  faceymin; faceboxes[i][30] =  facezmin; faceboxes[i][31] = 1.0f;
    }
}

void obj::setColor(glm::vec3 newColor){
	cbosize = ibosize*3;
  cbo_vec.clear();
  cbo_vec.resize(cbosize);

	for(int i=0; i<(cbosize/3); i++){
		int j = i*3;
    cbo_vec[j+0] = newColor[0]; cbo_vec[j+1] = newColor[1]; cbo_vec[j+2] = newColor[2];
		//cbo[j+0] = newColor[0]; cbo[j+1] = newColor[1]; cbo[j+2] = newColor[2];
	}
	defaultColor = newColor;
}

std::vector<glm::vec4>* obj::getPoints(){
    return &points;
}

std::vector<std::vector<int> >* obj::getFaces(){
    return &faces;
}

std::vector<std::vector<int> >* obj::getFaceNormals(){
    return &facenormals;
}

std::vector<std::vector<int> >* obj::getFaceTextures(){
    return &facetextures;
}

std::vector<glm::vec4>* obj::getNormals(){
    return &normals;
}

std::vector<glm::vec4>* obj::getTextureCoords(){
    return &texturecoords;
}

std::vector<float*>* obj::getFaceBoxes(){
    return &faceboxes;
}

glm::vec3 obj::getColor(){
	return defaultColor;
}

const std::vector<float>& obj::getVBO() const
{
  return vbo_vec;
}

const std::vector<float>& obj::getCBO() const
{
  return cbo_vec;
}

const std::vector<float>& obj::getNBO() const
{
  return nbo_vec;
}

const std::vector<int>& obj::getIBO() const
{
  return ibo_vec;
}

const std::vector<float>& obj::getTBO() const
{
  return tbo_vec;
}

int obj::getVBOsize() const
{
	return vbosize;
}

int obj::getNBOsize() const
{
	return nbosize;
}

int obj::getIBOsize() const
{
	return ibosize;
}

int obj::getCBOsize() const
{
	return cbosize;
}

int obj::getTBOsize() const
{
  return tbosize;
}

