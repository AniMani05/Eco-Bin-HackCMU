const express = require('express');
const app = express();

// Define a route that serves an HTML page with a button
app.get('/', (req, res) => {
    res.send(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>Button Example</title>
        </head>
        <body>
            <button onclick="alert('Button Clicked!')">Click Me</button>
        </body>
        </html>
    `);
});

// Listen on port 3000
const port = 1000;
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});

