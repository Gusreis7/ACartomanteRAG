services:
  rustfs:
    image: rustfs/rustfs:1.0.0
    container_name: acartomante-rustfs

    depends_on:
      volume-permission-helper:
        condition: service_completed_successfully

    security_opt:
      - no-new-privileges:true

    ports:
      - "9000:9000"
      - "9001:9001"

    environment:
      RUSTFS_VOLUMES: /data/rustfs{0...3}
      RUSTFS_ADDRESS: 0.0.0.0:9000

      RUSTFS_CONSOLE_ADDRESS: 0.0.0.0:9001
      RUSTFS_CONSOLE_ENABLE: "true"

      RUSTFS_ACCESS_KEY: ${RUSTFS_ACCESS_KEY}
      RUSTFS_SECRET_KEY: ${RUSTFS_SECRET_KEY}

      RUSTFS_OBS_LOGGER_LEVEL: info

    volumes:
      - rustfs_data_0:/data/rustfs0
      - rustfs_data_1:/data/rustfs1
      - rustfs_data_2:/data/rustfs2
      - rustfs_data_3:/data/rustfs3

    healthcheck:
      test:
        [
          "CMD",
          "sh",
          "-ec",
          "curl -fsS http://127.0.0.1:9000/health && curl -fsS http://127.0.0.1:9001/rustfs/console/health"
        ]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 20s

    restart: unless-stopped


  volume-permission-helper:
    image: alpine:3.22

    volumes:
      - rustfs_data_0:/data0
      - rustfs_data_1:/data1
      - rustfs_data_2:/data2
      - rustfs_data_3:/data3

    command: >
      sh -c "
        chown -R 10001:10001
        /data0
        /data1
        /data2
        /data3
      "

    restart: "no"


volumes:
  rustfs_data_0:
  rustfs_data_1:
  rustfs_data_2:
  rustfs_data_3: