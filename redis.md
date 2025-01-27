# Redis

Redis is a powerful, in-memory data structure store used as a database, cache, and message broker. It is widely used for its high performance and flexibility in handling various use cases like caching, message brokering, and more.

---

## Step 1: What is Redis?

### Key-Value Store
Redis is a **key-value store**, meaning it stores data as pairs of keys and values. This is similar to dictionaries or hashmaps in programming, where you can quickly store and retrieve data using keys.

### Blazing Fast
Redis stores its data in memory (RAM), making it **extremely fast** for data access and retrieval. This is why Redis is often used for caching frequently accessed data.

### Use Cases:
- **Caching**: Store frequently accessed data like session data or user profiles to improve performance.
- **Rate Limiting**: Implement rate limiters to restrict the number of requests a user can make.
- **Queues**: Use Redis to manage message queues or job processing systems.
- **Real-Time Analytics**: Process and analyze real-time data streams.
- **Pub/Sub Messaging**: Implement Publish/Subscribe messaging patterns.

---

## Step 2: Basic Redis Commands

### 1. Storing and Retrieving Values

You can store and retrieve data using simple **SET** and **GET** commands.

```bash
SET key value        # Store a value
GET key              # Retrieve the value
```

**Example:**
```bash
SET name "Biswarup"
GET name
# Output: "Biswarup"
```

### 2. Deleting Keys

You can delete a key using the **DEL** command.

```bash
DEL key              # Delete a key
```

**Example:**
```bash
SET name "Biswarup"
DEL name
GET name
# Output: (nil)
```

### 3. Key Expiry

Redis allows setting an expiration time for keys. The **EX** option sets the expiry in seconds.

```bash
SET key value EX seconds    # Set key with expiration
TTL key                    # Check time-to-live (TTL)
```

**Example:**
```bash
SET session "active" EX 10   # Key expires in 10 seconds
TTL session                 # Output: 10, 9, 8... then (nil)
```

### 4. Increment/Decrement

Redis supports atomic operations to **increment** or **decrement** numeric values.

```bash
INCR key        # Increment value
DECR key        # Decrement value
```

**Example:**
```bash
SET counter 10
INCR counter   # Output: 11
DECR counter   # Output: 10
```

### 5. Storing Lists

Redis supports lists, allowing you to push values to the beginning or end of a list and retrieve them.

```bash
LPUSH list value        # Add value to the beginning of a list
RPUSH list value        # Add value to the end of a list
LRANGE list 0 -1        # Retrieve all items in the list
LPOP list               # Remove and return the first item
RPOP list               # Remove and return the last item
```

**Example:**
```bash
LPUSH colors "red"
LPUSH colors "blue"
LRANGE colors 0 -1
# Output: ["blue", "red"]
```

### 6. Storing Hashes

Redis hashes are like **objects** or **dictionaries**, where you can store field-value pairs.

```bash
HSET hash field value   # Set field-value pair
HGET hash field         # Get value of a field
HGETALL hash            # Get all fields and values
```

**Example:**
```bash
HSET user:1 name "Alice"
HSET user:1 age 25
HGET user:1 name
# Output: "Alice"
HGETALL user:1
# Output: ["name", "Alice", "age", "25"]
```

### 7. Storing Sets

Redis sets are unordered collections of **unique values**. You can add, remove, or check values in a set.

```bash
SADD set value         # Add value to set
SMEMBERS set           # Retrieve all members
SREM set value         # Remove value from set
```

**Example:**
```bash
SADD tags "javascript"
SADD tags "redis"
SMEMBERS tags
# Output: ["javascript", "redis"]
```

### 8. Expiry and TTL

Redis can automatically expire keys after a set time using the **EXPIRE** command.

```bash
EXPIRE key seconds     # Set expiration time
TTL key                # Get time left before expiration
```

---

## Step 3: Installing Redis

### Install Redis

- **On Linux**:
  ```bash
  sudo apt update
  sudo apt install redis
  ```

- **On macOS**:
  ```bash
  brew install redis
  ```

- **On Windows**:
  Use **Docker** or **WSL** to install Redis.

### Running Redis

To start the Redis server, use:
```bash
redis-server
```

### Testing Redis

Open another terminal window to test Redis using the **redis-cli**:
```bash
redis-cli
```

From there, you can run commands like `SET` and `GET`.

---

## Step 4: Redis in Node.js

### Install Redis Client in Node.js

Install the Redis client library for Node.js using npm:

```bash
npm install redis
```

### Connecting to Redis in Node.js

Here’s how to connect to Redis in your Node.js application:

```javascript
const Redis = require("redis");
const redisClient = Redis.createClient();
redisClient.connect();

redisClient.on("connect", () => console.log("Redis connected!"));
redisClient.on("error", (err) => console.error("Redis error:", err));
```

### Storing and Retrieving Data

```javascript
await redisClient.set("key", "value");
const value = await redisClient.get("key");
console.log("Value:", value);  // Output: value
```

---

## Step 5: Advanced Features

### Pub/Sub (Publish/Subscribe)

Pub/Sub allows you to publish messages to a channel, and other clients can subscribe to it.

#### Publisher (Sending Messages)
```javascript
const publisher = Redis.createClient();
publisher.connect();

publisher.publish("news", "Redis is awesome!");
```

#### Subscriber (Receiving Messages)
```javascript
const subscriber = Redis.createClient();
subscriber.connect();

subscriber.subscribe("news", (message) => {
  console.log("Received message:", message);
});
```

When a message is published on the "news" channel, the subscriber will receive it in real-time.

### Queues in Redis

You can implement queues using Redis lists. Producers add tasks to the queue, and consumers process them.

#### Producer (Adding to the Queue)
```javascript
const producer = Redis.createClient();
await producer.connect();
await producer.lPush("taskQueue", JSON.stringify({ taskId: 1, taskName: "Process Data" }));
```

#### Consumer (Processing from the Queue)
```javascript
const consumer = Redis.createClient();
await consumer.connect();

consumer.blPop("taskQueue", 0, (err, data) => {
  const task = JSON.parse(data[1]);
  console.log("Processing task:", task);
});
```

The `blPop` command blocks the consumer until there’s data in the queue, making it ideal for job processing systems.

### Streams

Redis Streams allow you to work with real-time data feeds, ideal for log processing or event-driven systems.

```javascript
// Producer writes to the stream
await redisClient.xAdd("mystream", "*", { userId: 1, action: "login" });

// Consumer reads from the stream
const entries = await redisClient.xRange("mystream");
console.log(entries);
```

### Transactions

Redis supports transactions that allow you to group multiple commands and execute them atomically.

```javascript
const transaction = redisClient.multi();
transaction.set("key1", "value1");
transaction.set("key2", "value2");
await transaction.exec();  // Executes all commands atomically
```

---

Redis is incredibly flexible, fast, and powerful for various use cases. Whether you're caching data, handling real-time messaging with Pub/Sub, or managing queues, Redis will help you scale your applications efficiently. Let me know if you'd like to dive deeper into any specific concepts!
```

This version includes both the theory and commands, with installation instructions, and provides a detailed breakdown of Redis features, including Node.js integration, Pub/Sub, queues, and advanced concepts like streams and transactions.
