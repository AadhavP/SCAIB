# Development image for the Vite/React console with hot reload.
# For production use, build static assets and serve them (e.g. via nginx).

FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
