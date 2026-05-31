package com.anomaly.service;

import com.anomaly.config.MinioConfig;
import io.minio.*;
import io.minio.http.Method;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
public class MinioService {

    @Autowired
    private MinioClient minioClient;

    @Autowired
    private MinioConfig minioConfig;

    /**
     * 判断 bucket 是否存在，不存在则创建
     */
    public void createBucketIfAbsent() throws Exception {
        boolean isExist = minioClient
                .bucketExists(BucketExistsArgs.builder().bucket(minioConfig.getBucketName()).build());
        if (!isExist) {
            minioClient.makeBucket(MakeBucketArgs.builder().bucket(minioConfig.getBucketName()).build());
        }

        // 公共读权限策略 - 修复: ListMultipartUploadParts 只能用在 object 级别
        String policy = "{\"Version\":\"2012-10-17\",\"Statement\":["
                + "{\"Effect\":\"Allow\",\"Principal\":{\"AWS\":[\"*\"]},\"Action\":[\"s3:GetBucketLocation\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::"
                + minioConfig.getBucketName()
                + "\"]},"
                + "{\"Effect\":\"Allow\",\"Principal\":{\"AWS\":[\"*\"]},\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\"],\"Resource\":[\"arn:aws:s3:::"
                + minioConfig.getBucketName() + "/*\"]}]}";
        minioClient.setBucketPolicy(
                SetBucketPolicyArgs.builder().bucket(minioConfig.getBucketName()).config(policy).build());
    }

    /**
     * 上传文件并返回一个持续可用(默认7天)的预签名URL
     */
    public String uploadFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            return null;
        }

        try {
            createBucketIfAbsent();

            // 生成唯一文件名
            String originalFileName = file.getOriginalFilename();
            String extension = "";
            if (originalFileName != null && originalFileName.contains(".")) {
                extension = originalFileName.substring(originalFileName.lastIndexOf("."));
            }
            String objectName = UUID.randomUUID().toString() + extension;

            // 上传文件流
            InputStream inputStream = file.getInputStream();
            minioClient.putObject(
                    PutObjectArgs.builder()
                            .bucket(minioConfig.getBucketName())
                            .object(objectName)
                            .stream(inputStream, file.getSize(), -1)
                            .contentType(file.getContentType())
                            .build());
            inputStream.close();

            // 由于已经开启了公有读政策，直接拼装全网通用的纯净下载路径，抛弃含有跨域签名报错隐患的 presignedUrl
            String publicUrl = minioConfig.getUrl() + "/" + minioConfig.getBucketName() + "/" + objectName;

            log.info("文件上传 MinIO 成功: {}, 纯净链接: {}", objectName, publicUrl);
            return publicUrl;

        } catch (Exception e) {
            log.error("MinIO 文件上传失败", e);
            throw new RuntimeException("文件上传失败: " + e.getMessage());
        }
    }
}
